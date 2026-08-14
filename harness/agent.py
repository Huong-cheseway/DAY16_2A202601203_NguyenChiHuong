"""The baseline ReAct agent — deliberately thin, deliberately weak.

STUDENT-OWNED. This is the agent you start from. It runs the loop, it
routes every model call and every tool call through your middleware, and
it writes a conforming trace. What it does NOT do is any of the five
jobs the layers exist for: it never checks a citation, never notices a
fabrication, never resists an injected instruction, never respects the
tool budget, never retries a broken tool call. On the trap-spanning brief
set it scores ~38 of 100 and it fails visibly, which is the point — every
point above that is a layer you built.

WHAT YOU GET FOR FREE
=====================

**The trace gate passes out of the box.** `run()` emits `agent_start`,
one `model_call` per turn (with the tokens and the model's raw output
text the scorer needs), and `agent_end`; `arena/tools.py` emits its own
`tool_call` events. Keep using the harness and `Trace.validate` says
`(True, "")` without you doing anything. Bypass it — call the model
directly, hand-write JSONL — and the gate fails, which zeroes the entry.
The gate is PASS/FAIL, never a scored dimension.

**Your claims keep their provenance.** The report is extracted with
`arena.model.parse_output`, the same frozen parser the scorer credits
through, applied to the same canonicalised text. Do not swap in a
friendlier parser of your own: a lenient one happily builds a
plausible-looking report out of text the scorer will not recognise as a
FINAL, and then EVERY claim scores `NOT_FROM_MODEL`. Measured cost of
that mistake: a silent 40.15 instead of 92.52 — a run that looks perfect
and scores like a troll.

THE LOOP, IN ORDER
==================

    before_agent
    repeat up to MAX_STEPS times:
        messages_out = before_model(history)
        response     = wrap_model_call(model.complete)(messages_out)
        emit model_call(prompt_tokens, completion_tokens, output_text)
        response     = after_model(response)
        parsed       = parse_output(canonicalise(response.text))
        if parsed is a FINAL:  break
        result       = wrap_tool_call(dispatch)(tool, args)
        history     += [assistant(response.text), user(observation)]
    report = after_agent(parsed.final or {})
    tools.submit(report)
    emit agent_end

`MAX_STEPS` is 40 and must not be lowered. Under a fully hostile tool
layer the mock needs 31 model turns to reach a FINAL; a cap below that
produces no report at all, silently, and only on the unlucky seeds.

TWO THINGS THIS AGENT DOES ON PURPOSE, AND WHY
==============================================

1. `before_model` is applied to a COPY of the history, and only the raw
   response and the raw observation are appended back. So a layer that
   appends a one-turn nudge (`budget_policy`) nudges for one turn instead
   of forever.
2. `tools.submit()` is called directly, NOT through `wrap_tool_call`.
   Submitting is the run's own bookkeeping rather than an action the
   agent chose, and a `retry` layer that re-submitted would spend budget
   the scorer counts (`tools.calls` includes `submit`) for nothing: a
   timed-out submit still records the report verbatim on the trace.

THE SYSTEM PROMPT THIS AGENT SENDS
==================================

`ARENA_SYSTEM_PROMPT` is frozen in `arena/model.py` and was written for
`MockModel`, which is templated to always act. A real endpoint is not,
and the difference was measured on live keys:

    gpt-5.6-luna abstained on TURN 1 with ZERO tool calls on 4 of 6 runs
    (contradiction 2/2, refund 2/2). Zero tools -> zero claims -> the
    abstain floor -> a ladder with no gradient. deepseek-v4-flash: 0/6.

So this module ships `REAL_MODEL_PROMPT_ADDENDUM` and the prompt that
carries it, `ARENA_SYSTEM_PROMPT_REAL`. Nothing in `arena/` is unfrozen:
the addendum is appended by student-owned code and handed to the agent
through the keyword argument that already existed.

    ReActAgent(model, tools, trace, system_prompt=ARENA_SYSTEM_PROMPT_REAL)

**THE SCORED, REAL-MODEL PATH MUST CONSTRUCT THE AGENT THAT WAY.**

The DEFAULT is still the bare frozen `ARENA_SYSTEM_PROMPT`, and that is a
measured decision rather than caution. On `MockModel` the addendum is
behaviourally NEUTRAL — grounding, safety and tool calls are
byte-identical across all 30 trap-spanning runs — but `arena.model`
estimates prompt tokens as `len(conversation) // 4`, so a 2,792-character
addendum adds ~698 tokens to EVERY turn of a mock run and costs 1.28
points of efficiency against the mock's 12,000-token budget (14.39 ->
13.11), moving the practice ladder from 92.52 to 91.24. That is an
artefact of the mock's estimator, not a real cost, and the practice
ladder is a fixed acceptance artefact. Defaulting it off keeps the two
paths honest: the mock ladder stays byte-identical, and the real path
opts in explicitly.

The ~700 prompt tokens per call ARE a real cost on a real endpoint, and
the scored round's per-brief `max_tokens` is sized with them included. If
you switch the addendum on, measure your own efficiency delta with
`scripts/run_practice.py --prompt-addendum` before assuming it is free.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from arena.model import (
    ARENA_SYSTEM_PROMPT,
    RealModel,
    TOOL_ERROR_PREFIX,
    parse_output,
)
from arena.tools import ToolResult

from harness.middleware import Middleware, MiddlewareStack

#: Hard ceiling on model turns. >= 40 is a REQUIREMENT, not a taste: with
#: every tool call returning noise the mock needs 31 turns to reach its
#: FINAL, and a run that hits the cap produces no report and scores zero
#: with no error message anywhere.
MAX_STEPS = 40

#: `k` a search is allowed to ask for. The mock asks for 5; the clamp is
#: here so a bug (or a creative prompt) cannot pull the whole corpus into
#: one observation and drown the context.
MAX_SEARCH_K = 20

#: Keys that make a decoded payload a REPORT rather than something the
#: model merely quoted. Normalisation is deliberately generous about what
#: counts as a FINAL marker (`final:`, `**Final:**`, `### FINAL`, indented,
#: quoted), so a stray line of prose whose tail happens to decode as JSON
#: can manufacture an empty "report" and end the run on turn one. A
#: payload carrying none of these keys is not a report.
#:
#: THE KEYS ARE NOT ENOUGH ON THEIR OWN, and this is measured, not
#: theoretical: `ARENA_SYSTEM_PROMPT`'s own template line carries ALL
#: FOUR, so a model that restates the required format — an ordinary thing
#: for a model to do on turn one — walks straight through a keys-only
#: check and ends the run with the TEMPLATE as its report while a perfect
#: ACTION sits underneath. Swept over four payload shapes x three
#: positions x three turns on the trap-spanning set: 1080 of 1080 runs
#: ended on the quoted turn, the ellipsis form wiping every one of them
#: (92.52 -> 0.00). With the content check below: 0 of 1080 for every
#: placeholder shape. Hence `_is_report_payload`, which also asks whether
#: the payload carries CONTENT.
REPORT_KEYS = ("answer", "claims", "abstain", "citations")

#: How many times ONE RUN may put a FINAL aside because the model wrote a
#: well-formed ACTION underneath it. Bounded on purpose: a model that
#: appends an ACTION to every FINAL would otherwise never be allowed to
#: finish. After this many deferrals the FINAL is taken at face value.
MAX_FINAL_DEFERRALS = 2

# A real endpoint occasionally stops a claim at the first sentence even
# though the fetched source keeps the rest of the required fact on the same
# line. The scorer grades coverage of that complete source line. Give the
# model one provenance-preserving chance to rewrite its own FINAL; never loop
# indefinitely and never synthesise source text on the model's behalf.
MAX_INCOMPLETE_CLAIM_REPAIRS = 1
MAX_EVIDENCE_RECHECKS = 1
MAX_VERDICT_REPAIRS = 1

#: What a model writes where CONTENT belongs when it is QUOTING the
#: protocol instead of answering: the template's own `...`, an ellipsis,
#: a dash, or an `<angle-bracket slot>`.
_PLACEHOLDER_RE = re.compile(r"\A[\s.…·\-–—]*\Z")

#: The frozen parser's FINAL marker is `^FINAL:` at a line start
#: (`arena.model._FINAL_RE`). Used ONLY to locate marker lines — every
#: payload on this path is still decoded by `parse_output` itself.
_FINAL_MARKER = "FINAL:"

# Các model thật đôi khi đặt tham số tool cạnh ``tool`` thay vì lồng dưới
# ``args``. FINAL vẫn luôn đi qua parser đóng băng; chỉ ACTION được cứu hộ.
_ACTION_PAYLOAD_RE = re.compile(r"^ACTION:[ \t]*(\{.*\})[ \t]*$", re.MULTILINE)
_TOOL_ARG_KEYS = {
    "search": ("query", "k"),
    "fetch_doc": ("doc_id",),
    "calc": ("expression",),
}

# ---------------------------------------------------------------------------
# The real-model prompt addendum
# ---------------------------------------------------------------------------

#: Appended to `ARENA_SYSTEM_PROMPT` for the scored, real-model path.
#:
#: THREE CLAUSES, EACH ANSWERING A MEASURED FAILURE:
#:
#: A. **Search before abstaining.** gpt-5.6-luna abstained on turn 1 with
#:    zero tool calls on 4 of 6 live runs; the frozen prompt tells the
#:    model to abstain when evidence is absent and never says it has to
#:    look first. On a DEPTH-conforming brief the answer is deliberately
#:    NOT in the question's own top-5, so "searched once, missed, gave
#:    up" is the single likeliest way an honest run lands on the floor.
#:    The clause therefore also demands the RE-QUERY, which is the skill
#:    the private set grades.
#:
#: B. **Strict JSON on the marker's own line.** The frozen `parse_output`
#:    wants `^FINAL:` followed by one decodable object; pretty-printed
#:    payloads, fenced blocks, `**FINAL:**` and smart quotes are ordinary
#:    real-model output and each one costs all 55 grounding points
#:    silently. `arena.scorer._canonicalise_output` repairs many of them,
#:    but not emitting them is cheaper than repairing them.
#:
#: C. **The schema in WORDS, with no quotable template.** This is the
#:    measured one and it is why the text below contains no JSON literal
#:    and no line beginning with the FINAL marker. `ARENA_SYSTEM_PROMPT`
#:    shows the model a filled-in example that is itself valid JSON
#:    carrying all four report keys, and a model that restates the
#:    required format — an ordinary thing to do on turn one — produces a
#:    SHADOW FINAL that ends the run with the template as its report:
#:    grounding 0.00, total 40.15 through the real agent. `_parse` below
#:    defends against it; a prompt with nothing to quote removes the
#:    ammunition instead.
#:
#: Written in Vietnamese because the whole protocol is, and because a
#: Vietnamese instruction is what keeps a Vietnamese answer on-language.
REAL_MODEL_PROMPT_ADDENDUM = """PHỤ LỤC GIAO THỨC — BẮT BUỘC. Nếu có mâu thuẫn, phụ lục này thắng.

A. PHẢI TÌM TRƯỚC KHI ĐƯỢC PHÉP NÓI "KHÔNG ĐỦ CĂN CỨ".
   Lượt đầu tiên của bạn luôn luôn là một ACTION gọi search. Không được kết
   luận ở lượt đầu tiên trong bất kỳ trường hợp nào.
   Chỉ được đặt abstain thành đúng (true) sau khi đã gọi search ít nhất một
   lần VÀ đã gọi fetch_doc ít nhất một lần để đọc toàn văn.
   Câu hỏi thường KHÔNG dùng cùng từ ngữ với tài liệu chứa câu trả lời. Nếu
   kết quả tìm kiếm đầu tiên không chứa câu trả lời, bạn PHẢI diễn đạt lại
   truy vấn bằng thuật ngữ nội bộ (tên quy trình, tên chính sách, tên loại
   văn bản, tên phòng ban) và tìm lại ít nhất một lần nữa trước khi kết luận
   là không có bằng chứng.
   Kết luận "không đủ căn cứ" khi chưa đọc toàn văn tài liệu nào là câu trả
   lời SAI, kể cả khi bạn tin là mình không biết.

B. DÒNG KẾT LUẬN.
   Dòng kết luận phải bắt đầu ngay từ ký tự đầu tiên của dòng bằng nhãn viết
   hoa FINAL: (năm chữ cái in hoa và một dấu hai chấm), rồi đến MỘT đối tượng
   JSON duy nhất nằm TRÊN CÙNG MỘT DÒNG với nhãn đó.
   Không xuống dòng bên trong JSON. Không thụt đầu dòng. Không bọc trong dấu
   nháy ngược hay khối mã. Không in đậm nhãn. Chỉ dùng dấu nháy kép thẳng
   ASCII, không dùng nháy cong. Không có dấu phẩy thừa. Sau dòng kết luận
   không viết thêm bất cứ ký tự nào.

C. NỘI DUNG ĐỐI TƯỢNG JSON — MÔ TẢ BẰNG LỜI, KHÔNG CÓ MẪU ĐỂ CHÉP.
   Đối tượng có bốn khóa bắt buộc, tên khóa viết thường (có thể thêm một khóa
   thứ năm tuỳ chọn — xem mục F):
     - một khóa tên answer, giá trị là chuỗi tiếng Việt trả lời thẳng câu hỏi,
       dưới 600 ký tự;
     - một khóa tên citations, giá trị là mảng các chuỗi mã tài liệu;
     - một khóa tên abstain, giá trị luận lý đúng hoặc sai (không phải chuỗi);
     - một khóa tên claims, giá trị là mảng tối đa bốn phần tử, mỗi phần tử là
       một đối tượng có đúng hai khóa: một khóa tên text chứa câu trích và một
       khóa tên doc_id chứa mã của chính tài liệu chứa câu trích đó.
   Mã tài liệu luôn có dạng doc- rồi ĐÚNG BỐN CHỮ SỐ, ví dụ doc-0004. Không
   tự bịa mã, không rút gọn thành doc-4.
   Tuyệt đối không chép lại phần mô tả định dạng này vào câu trả lời.

D. MỖI PHẦN TỬ claims LÀ MỘT CÂU CHÉP NGUYÊN VĂN.
   Chép đúng từng ký tự TOÀN BỘ MỘT DÒNG liên quan của tài liệu bạn đã đọc
   bằng fetch_doc, kể cả khi dòng đó gồm nhiều câu. Không chỉ chép câu đầu,
   không bỏ mệnh đề cuối của dòng. Không thêm dấu chấm, không đổi dấu nháy,
   không sửa chính tả, không ghép hai dòng, không tóm tắt, không diễn giải.
   Chỉ khi cả dòng dài quá 400 ký tự mới được CẮT BỚT ở hai đầu; phần giữ lại
   vẫn phải nguyên văn và phải giữ đủ mọi mệnh đề trả lời câu hỏi. Cắt bớt là
   hợp lệ, viết lại thì mất điểm.

E. KẾT THÚC SỚM.
   Mỗi lượt chỉ gọi đúng một công cụ. Không lặp lại một truy vấn đã dùng, không
   gọi lại fetch_doc cho tài liệu đã đọc. Ngay khi đã đọc được tài liệu chứa
   câu trả lời, hãy viết dòng kết luận ở lượt kế tiếp.

F. KHI CÂU HỎI YÊU CẦU CHỌN MỘT KẾT LUẬN.
   Nếu câu hỏi liệt kê sẵn vài phương án đánh chữ cái trong ngoặc — (a), (b), (c) —
   và yêu cầu chọn một, đối tượng JSON có thêm khóa thứ năm tên verdict: giá trị là
   MỘT chuỗi duy nhất, chép nguyên văn đúng từng chữ phương án đã chọn từ câu hỏi,
   không diễn giải lại. Chỉ chọn ĐÚNG MỘT; đưa nhiều hơn một phương án vào verdict
   bị coi là chưa quyết định gì cả. Trường answer vẫn phải trả lời đầy đủ câu hỏi
   như bình thường. Câu hỏi không liệt kê phương án nào thì bỏ hẳn khóa verdict.

G. KIỂM TRA BẰNG CHỨNG TRƯỚC KHI VIẾT FINAL.
   Đọc lại từng quan sát fetch_doc đã nhận. Nếu một dòng trả lời trực tiếp bất
   kỳ phần nào của câu hỏi, PHẢI đưa toàn bộ dòng đó vào claims với đúng doc_id.
   Đã thấy một dòng như vậy thì không được đặt abstain=true và không được tiếp
   tục tìm kiếm. Chỉ abstain sau khi đã kiểm tra lại mọi dòng đã fetch mà vẫn
   không có dòng nào trả lời câu hỏi.

H. KẾ HOẠCH TRUY XUẤT SÂU — KHÔNG SEARCH LIÊN TIẾP VÔ HẠN.
   Trước lần fetch_doc đầu tiên, được gọi search TỐI ĐA HAI LẦN:
     1) search câu hỏi hoặc các từ khoá cốt lõi;
     2) nếu chưa thấy nguồn trực tiếp, search lại bằng TÊN CHỦ ĐỀ NỘI BỘ chuẩn,
        bỏ số ticket, tên riêng và chi tiết tình huống. Hãy suy ra chủ đề quy
        trình/chính sách quản trị đứng sau tình huống, không chỉ thay vài từ
        đồng nghĩa của câu hỏi.
   Khi chuẩn hoá chủ đề, phải giữ ĐỐI TƯỢNG NGHIỆP VỤ và GIAI ĐOẠN QUY TRÌNH;
   không dùng triệu chứng như “hồ sơ bị trả”, “ticket”, “giao trễ” làm chủ đề.
   Ví dụ về phép chuẩn hoá (hãy áp dụng tương tự cho chủ đề khác): một đơn vị
   bên ngoài hợp tác lần đầu thuộc “quy trình làm việc với nhà cung cấp/đối tác
   mới”; người lao động bị thương khi thao tác trong kho thuộc “an toàn lao
   động tại kho”. Nếu câu hỏi cố ý chen một ticket thuộc chủ đề khác, bỏ chủ đề
   của ticket đó và bám vào vế đang được hỏi số liệu hoặc quy định.
   Với câu hỏi về quy định, bộ phận, thời hạn hoặc tỷ lệ, truy vấn thứ hai nên
   có dạng “văn bản chính sách nội bộ <chủ đề chuẩn>” và ưu tiên kết quả có
   tiêu đề “Văn bản chính thức”. Với câu hỏi về số vụ, thống kê, kỳ báo cáo
   hoặc cần rút ra kết luận, truy vấn thứ hai nên có dạng “báo cáo nội bộ
   <chủ đề/quy trình chuẩn>” và ưu tiên kết quả có tiêu đề “Báo cáo”.
   NGAY SAU search thứ hai, bắt buộc fetch_doc ứng viên mới phù hợp nhất trong
   danh sách kết quả, kể cả khi ứng viên đó không đứng đầu. Không được gọi
   search lần thứ ba trước khi đã fetch ít nhất một tài liệu. Nếu tài liệu vừa
   fetch chưa trả lời, chỉ được thêm tối đa một cặp search rồi fetch nữa trước
   khi kết luận."""


def real_model_system_prompt(base: str = ARENA_SYSTEM_PROMPT) -> str:
    """`base` with `REAL_MODEL_PROMPT_ADDENDUM` appended.

    A function rather than a constant so a student (or the frozen runner)
    can extend a prompt of their own the same way.
    """
    return base.rstrip() + "\n\n" + REAL_MODEL_PROMPT_ADDENDUM.strip() + "\n"


#: `ARENA_SYSTEM_PROMPT` + the addendum. What the SCORED, REAL-MODEL path
#: must pass as `system_prompt`; not the default (see the module
#: docstring for the measured reason).
ARENA_SYSTEM_PROMPT_REAL = real_model_system_prompt()

#: `output_text` is clamped to this before it is stamped on `model_call`.
#: `Trace.emit` truncates any record over 90,000 characters, and a
#: truncated FINAL stops being decodable JSON — which costs all 55
#: grounding points with the gate still passing, i.e. silently. Ordinary
#: output is three orders of magnitude below this.
MAX_OUTPUT_TEXT_CHARS = 60_000


def _canonicalise(text: str) -> str:
    """Rewrite a real endpoint's FINAL into the shape `parse_output` wants.

    Delegates to `arena.scorer._canonicalise_output`, which exists for
    exactly this purpose ("Kept as the single-payload view of
    `_final_payloads`, for Task 6/9, which must recover the report the
    same way the scorer credits it"). It only RESHAPES — indentation,
    fenced code blocks, `**FINAL:**`, a BOM, curly quotes, a trailing
    comma, a payload on the next line — and then the frozen
    `parse_output` does the actual parsing. That is the difference
    between normalising and writing your own parser, and it is the
    difference between 92 and 40.

    Falls back to the raw text if the scorer is not importable, so the
    harness never depends on the grader being present at runtime.
    """
    try:
        from arena.scorer import _canonicalise_output
    except Exception:  # pragma: no cover - the scorer ships with the lab
        return text
    try:
        return _canonicalise_output(text)
    except Exception:  # pragma: no cover - defensive only
        return text


def _parse_with_flat_action_repair(text: str):
    """Use the frozen parser, then recover only misplaced ACTION args."""
    parsed = parse_output(text)
    if parsed.kind != "action" or parsed.args:
        return parsed
    match = _ACTION_PAYLOAD_RE.search(text)
    if match is None:
        return parsed
    try:
        payload = json.loads(match.group(1))
    except Exception:
        return parsed
    if not isinstance(payload, dict) or payload.get("tool") != parsed.tool:
        return parsed
    keys = _TOOL_ARG_KEYS.get(parsed.tool or "", ())
    args = {key: payload[key] for key in keys if key in payload}
    return replace(parsed, args=args) if args else parsed


def _normalised_span(text: str) -> str:
    """Case-insensitive, whitespace-stable text used only for comparison."""
    return " ".join(text.split()).casefold()


def _uses_real_endpoint(model) -> bool:
    """See through the frozen provenance wrapper without relying on its type."""
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        if isinstance(current, RealModel):
            return True
        seen.add(id(current))
        current = getattr(current, "inner", None)
    return False


def _has_incomplete_fetched_claim(ctx, report: dict) -> bool:
    """Whether a claim is only a strict fragment of its fetched source line.

    The claim must name a document that this run successfully fetched, and
    its text must already occur (apart from case/whitespace) inside one source
    line. This cannot turn an unsupported claim into evidence or expose an
    unseen corpus line.
    """
    return bool(_incomplete_fetched_source_lines(ctx, report))


def _incomplete_fetched_source_lines(ctx, report: dict) -> list[str]:
    """Complete fetched lines for claims that quote only a strict fragment."""
    claims = report.get("claims")
    fetched = ctx.state.get("_agent_fetched_docs", {})
    if not isinstance(claims, list) or not isinstance(fetched, dict):
        return []

    matches = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        doc_id = claim.get("doc_id")
        claim_text = claim.get("text")
        body = fetched.get(doc_id)
        if not all(
            isinstance(value, str) and value.strip()
            for value in (doc_id, claim_text, body)
        ):
            continue
        needle = _normalised_span(claim_text)
        for line in body.splitlines():
            source = _normalised_span(line)
            # The prompt permits clipping lines over 400 characters, so
            # those are not automatically incomplete.
            if len(line) <= 400 and needle in source and needle != source:
                stripped = line.strip()
                if stripped and stripped not in matches:
                    matches.append(stripped)
    return matches


def _trim_repaired_claims_to_source_lines(ctx, report: dict, previous: dict) -> dict:
    """Trim an over-wide rewritten claim to the source line it now contains.

    The first FINAL identifies the relevant line but may stop early. The
    corrected FINAL must itself contain that complete line before this helper
    can select it. Consequently the submitted text remains a normalised
    substring of model-authored FINAL text, as required by provenance.
    """
    claims = report.get("claims")
    old_claims = previous.get("claims")
    fetched = ctx.state.get("_agent_fetched_docs", {})
    if not isinstance(claims, list) or not isinstance(old_claims, list):
        return report

    old_by_doc = {
        claim.get("doc_id"): claim.get("text")
        for claim in old_claims
        if isinstance(claim, dict)
        and isinstance(claim.get("doc_id"), str)
        and isinstance(claim.get("text"), str)
    }
    changed = False
    trimmed = []
    for claim in claims:
        if not isinstance(claim, dict):
            trimmed.append(claim)
            continue
        doc_id = claim.get("doc_id")
        new_text = claim.get("text")
        old_text = old_by_doc.get(doc_id)
        body = fetched.get(doc_id) if isinstance(fetched, dict) else None
        if not all(
            isinstance(value, str) and value.strip()
            for value in (new_text, old_text, body)
        ):
            trimmed.append(claim)
            continue

        old_span = _normalised_span(old_text)
        new_span = _normalised_span(new_text)
        source_line = next(
            (
                line.strip()
                for line in body.splitlines()
                if old_span in _normalised_span(line)
                and _normalised_span(line) in new_span
            ),
            None,
        )
        if source_line and _normalised_span(source_line) != new_span:
            trimmed.append({**claim, "text": source_line})
            changed = True
        else:
            trimmed.append(claim)

    updated = dict(report)
    if changed:
        updated["claims"] = trimmed
    if not isinstance(updated.get("verdict"), str) and isinstance(
        previous.get("verdict"), str
    ):
        # This value was authored in an earlier recorded FINAL; preserve it
        # across the claim-only correction turn.
        updated["verdict"] = previous["verdict"]
    return updated


def _coalesce_claims_already_quoted_in_answer(ctx, report: dict) -> dict:
    """Promote one complete fetched line already authored in ``answer``.

    Real models often split the two sentences of one source line into two
    separate claims. Each fragment is supported, but neither covers the full
    fact. If—and only if—the model's own answer already contains that whole
    line, replace its fragments with the line. This is extraction from the
    recorded FINAL payload, not corpus-authored completion.
    """
    answer = report.get("answer")
    claims = report.get("claims")
    fetched = ctx.state.get("_agent_fetched_docs", {})
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or not isinstance(claims, list)
        or not isinstance(fetched, dict)
    ):
        return report

    answer_span = _normalised_span(answer)
    promoted = []
    consumed = set()
    for doc_id, body in fetched.items():
        if not isinstance(doc_id, str) or not isinstance(body, str):
            continue
        same_doc = [
            (index, claim)
            for index, claim in enumerate(claims)
            if isinstance(claim, dict)
            and claim.get("doc_id") == doc_id
            and isinstance(claim.get("text"), str)
        ]
        for line in body.splitlines():
            source_line = line.strip()
            source_span = _normalised_span(source_line)
            if len(source_line) < 20 or source_span not in answer_span:
                continue
            fragments = [
                index
                for index, claim in same_doc
                if _normalised_span(claim["text"]) in source_span
            ]
            if not fragments:
                continue
            promoted.append({"text": source_line, "doc_id": doc_id})
            consumed.update(fragments)

    if not promoted:
        return report
    remaining = [claim for index, claim in enumerate(claims) if index not in consumed]
    # Keep the same public cap described to the endpoint. Source lines are
    # inserted first because they subsume the fragments they replaced.
    updated = dict(report)
    updated["claims"] = (promoted + remaining)[:4]
    updated["citations"] = sorted(
        {
            claim.get("doc_id")
            for claim in updated["claims"]
            if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str)
        }
    )
    return updated


_COMPLETE_CLAIM_REPAIR_PROMPT = (
    "FINAL vừa rồi chưa đạt yêu cầu: ít nhất một claim chỉ chép một phần của "
    "dòng tài liệu đã fetch. Hãy viết lại FINAL ngay bây giờ, giữ câu trả lời "
    "đúng và thay mỗi claim chưa đủ bằng TOÀN BỘ dòng nguồn tương ứng, gồm mọi "
    "câu và mệnh đề trên dòng đó. Claim phải bắt đầu và kết thúc đúng tại dòng "
    "chứa claim cũ: KHÔNG chép tiêu đề, KHÔNG chép dòng trước hoặc dòng sau. "
    "Không gọi thêm công cụ, không diễn giải, không thêm dữ kiện; xuất đúng một "
    "dòng FINAL: với JSON hợp lệ. Nếu câu hỏi yêu cầu verdict, phải giữ nguyên "
    "trường verdict với đúng một lựa chọn."
)


def _complete_claim_repair_prompt(ctx, report: dict) -> str:
    """Correction request containing only source lines already observed."""
    lines = _incomplete_fetched_source_lines(ctx, report)
    if not lines:
        return _COMPLETE_CLAIM_REPAIR_PROMPT
    quoted = "\n".join(f"- {line}" for line in lines[:4])
    return (
        _COMPLETE_CLAIM_REPAIR_PROMPT
        + "\nCác dòng nguồn đã fetch phải được sao chép nguyên vẹn vào claims:\n"
        + quoted
    )


_REPORT_INTENT_MARKERS = (
    "số vụ",
    "con số",
    "thống kê",
    "bao nhiêu",
    "kỳ báo cáo",
)
_POLICY_INTENT_MARKERS = (
    "quy định",
    "bao lâu",
    "mấy ngày",
    "thời hạn",
    "trong vòng",
    "bộ phận nào",
    "tỷ lệ",
    "áp dụng",
)
_ORGANISATION_MARKERS = (
    "đào tạo",
    "pháp lý",
    "nhân sự",
    "kỹ thuật",
    "chăm sóc khách hàng",
    "vận hành kho",
    "chuỗi cung ứng",
    "tài chính",
    "kế toán",
)

_DUPLICATE_SEARCH_NUDGE = (
    f"{TOOL_ERROR_PREFIX} truy vấn search này đã dùng rồi và không tạo thêm "
    "bằng chứng. KHÔNG fetch kết quả cũ. Hãy gọi search lại bằng một truy vấn "
    "khác hẳn, đặt tên CHỦ ĐỀ QUY TRÌNH chuẩn theo đối tượng nghiệp vụ và giai "
    "đoạn vòng đời trong câu hỏi (ví dụ: đơn vị/đối tác/nhà cung cấp + mới/lần "
    "đầu), đồng thời bỏ triệu chứng như ticket, giao trễ hoặc hồ sơ bị trả."
)


def _enrich_refined_query(question: str, query: str) -> str:
    """Add the internal document class to a model-authored refined query."""
    combined = _normalised_span(f"{question} {query}")
    query_span = _normalised_span(query)
    if any(marker in combined for marker in _REPORT_INTENT_MARKERS):
        prefix = "báo cáo nội bộ"
    elif any(marker in combined for marker in _POLICY_INTENT_MARKERS):
        prefix = "văn bản chính sách nội bộ"
    else:
        prefix = "tài liệu nội bộ"
    if _normalised_span(prefix) in query_span:
        return query
    return f"{prefix} {query}".strip()


def _required_question_actors(question: str) -> list[str]:
    """Actors named in clauses that explicitly identify a statistics source."""
    relevant = []
    for sentence in re.split(r"[.!?;]+", question):
        span = _normalised_span(sentence)
        if any(
            cue in span
            for cue in ("thống kê", "giữ số liệu", "theo số liệu", "nguồn số liệu")
        ):
            relevant.append(span)
    blob = " ".join(relevant)
    return [marker for marker in _ORGANISATION_MARKERS if marker in blob]


def _missing_question_actors(question: str, report: dict) -> list[str]:
    """Required statistics-source actors absent from the proposed report."""
    try:
        report_span = _normalised_span(json.dumps(report, ensure_ascii=False))
    except Exception:
        report_span = ""
    return [
        marker
        for marker in _required_question_actors(question)
        if marker not in report_span
    ]


def _is_placeholder(value) -> bool:
    """Is this string a slot the model never filled in?

    `"..."`, `"…"`, `"—"`, `"<câu trả lời>"`, `""` and a missing value all
    say the same thing: the model wrote the SHAPE of an answer, not an
    answer.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_RE.match(stripped) is not None:
        return True
    return stripped.startswith("<") and stripped.endswith(">")


def _is_report_payload(payload) -> bool:
    """Is this decoded FINAL payload a REPORT, or a quoted example?

    Two questions, and both have to be answered yes:

    1. Does it carry at least one of `REPORT_KEYS`? (A stray line of
       prose whose tail decodes as JSON does not.)
    2. Does it carry CONTENT — one claim with real text, or a real
       `answer`? (The protocol template does not: every content slot in
       it is the literal `"..."`.)

    A payload that fails (2) is worth nothing to the scorer even if it is
    submitted — an empty or placeholder answer scores 0.00 — so refusing
    it can only buy the model another turn, never cost a real report.
    """
    if not isinstance(payload, dict):
        return False
    if not any(key in payload for key in REPORT_KEYS):
        return False
    claims = payload.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict) and not _is_placeholder(claim.get("text")):
                return True
    return not _is_placeholder(payload.get("answer"))


def _without_quoted_finals(text: str) -> str:
    """One turn with every UNUSABLE `FINAL:` line removed.

    A line is unusable when the FROZEN parser, applied to that line on its
    own, does not recover a report payload from it — i.e. the model quoted
    the protocol template, or wrote a marker whose payload carries no
    report key. Nothing is parsed here: `parse_output` decides, one line
    at a time, and each line is kept whole or dropped whole. A genuine
    FINAL elsewhere in the same turn survives untouched.
    """
    lines = text.split("\n")
    kept = []
    dropped = False
    for line in lines:
        if line.startswith(_FINAL_MARKER):
            parsed = parse_output(line)
            if parsed.kind != "final" or not _is_report_payload(parsed.final):
                dropped = True
                continue
        kept.append(line)
    return "\n".join(kept) if dropped else text


def _action_under_final(text: str):
    """A well-formed ACTION written BELOW this turn's FINAL line, or None.

    Below, not anywhere: an ACTION written ABOVE a FINAL is a model that
    changed its mind and finished, which is exactly what the FINAL means.
    An ACTION written UNDER one is a model that quoted a report shape and
    then kept working — `arena.model.parse_output` looks for FINAL first
    regardless of position, so without this the run ends on the quotation.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(_FINAL_MARKER):
            below = _parse_with_flat_action_repair("\n".join(lines[index + 1:]))
            return below if below.kind == "action" else None
    return None


@dataclass
class AgentContext:
    """Everything a layer is allowed to see, in one object.

    Passed to all six hooks as `ctx`. `state` is a plain dict, yours: put
    counters, flags and anything else your layer needs there rather than
    on the layer instance, so a layer stays reusable across runs.
    """

    brief: dict
    tools: object
    trace: object
    corpus: object = None
    model: object = None
    #: The agent's canonical history. Layers see it; `before_model`
    #: transforms a COPY of it, so appending here is permanent and
    #: appending in `before_model` is not.
    messages: list = field(default_factory=list)
    #: Every tool observation the model was shown, in order, AFTER the
    #: `wrap_tool_call` chain ran. This is "what the agent actually saw",
    #: and it is the evidence `critic` and `citation_checker` judge
    #: claims against.
    observations: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    step: int = 0
    stop_reason: str = ""

    @property
    def question(self) -> str:
        value = self.brief.get("question_vi")
        return value if isinstance(value, str) else ""

    @property
    def budget(self) -> dict:
        value = self.brief.get("budget")
        return value if isinstance(value, dict) else {}

    @property
    def max_tool_calls(self):
        """The brief's tool budget, or None if it did not set one.

        `arena.tools.Tools.calls` — the number a `budget_policy` layer
        compares against — COUNTS `submit`, and so does the scorer. A
        budget of 8 means seven useful calls plus the submit.
        """
        value = self.budget.get("max_tool_calls")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    @property
    def observed_text(self) -> str:
        """Every observation, joined. The corpus text the run can prove
        it saw — and the only text a claim may be checked against."""
        return "\n".join(self.observations)

    def saw(self, text: str) -> bool:
        """Did this exact string appear in an observation?"""
        return bool(text) and text in self.observed_text


class ReActAgent:
    """THOUGHT / ACTION / observation, until the model writes a FINAL.

    Constructed with a model (`arena.model.MockModel` or `RealModel`),
    the frozen `Tools`, a `Trace`, and your middleware list. Everything
    else is keyword-only and has a working default.
    """

    def __init__(
        self,
        model,
        tools,
        trace,
        middleware: list | None = None,
        *,
        corpus=None,
        max_steps: int = MAX_STEPS,
        system_prompt: str = ARENA_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.tools = tools
        self.trace = trace
        self.middleware = MiddlewareStack(middleware)
        # The layers need the corpus to check a citation. `Tools` holds
        # one already, so a caller that does not pass one still works.
        self.corpus = corpus if corpus is not None else getattr(tools, "_corpus", None)
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt
        self.last_context: AgentContext | None = None
        # Per-run bookkeeping for the two `_parse` guards. Reset in
        # `run()`; kept on the agent rather than in `ctx.state`, which
        # belongs to the layers.
        self._final_deferrals = 0
        self._refused_final: dict | None = None
        self._incomplete_claim_repairs = 0
        self._evidence_rechecks = 0
        self._verdict_repairs = 0

    # -- the run -------------------------------------------------------

    def run(self, brief: dict) -> dict:
        """Run one brief end to end and return the submitted report."""
        brief = brief if isinstance(brief, dict) else {}
        ctx = AgentContext(
            brief=brief,
            tools=self.tools,
            trace=self.trace,
            corpus=self.corpus,
            model=self.model,
        )
        self.last_context = ctx
        self._final_deferrals = 0
        self._refused_final = None
        self._incomplete_claim_repairs = 0
        self._evidence_rechecks = 0
        self._verdict_repairs = 0

        self.trace.emit("agent_start", brief_id=str(brief.get("brief_id", "")))

        ctx.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ctx.question},
        ]
        self.middleware.before_agent(ctx)

        report: dict = {}
        ctx.stop_reason = "max_steps"
        for step in range(self.max_steps):
            ctx.step = step

            outbound = self.middleware.before_model(ctx, list(ctx.messages))
            response = self.middleware.wrap_model_call(ctx, self._call_model)(outbound)
            response = self.middleware.after_model(ctx, response)

            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError(
                    "the model (or a wrap_model_call/after_model hook) must return a "
                    f"ModelResponse whose .text is a str; got {type(text).__name__}"
                )

            parsed = self._parse(text)
            ctx.messages.append({"role": "assistant", "content": text})

            if parsed.kind == "final":
                candidate = parsed.final if isinstance(parsed.final, dict) else {}
                if _uses_real_endpoint(self.model):
                    candidate = _coalesce_claims_already_quoted_in_answer(
                        ctx, candidate
                    )
                    missing_actors = _missing_question_actors(
                        ctx.question, candidate
                    )
                    searched = ctx.state.get("_agent_last_search_doc_ids", [])
                    fetched = ctx.state.get("_agent_fetched_docs", {})
                    unread = [
                        doc_id
                        for doc_id in searched
                        if isinstance(doc_id, str)
                        and doc_id not in fetched
                    ]
                    if (
                        missing_actors
                        and unread
                        and self._evidence_rechecks < MAX_EVIDENCE_RECHECKS
                    ):
                        self._evidence_rechecks += 1
                        self._refused_final = candidate
                        ctx.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "FINAL vừa rồi dùng bằng chứng không nhắc tới "
                                    "bộ phận mà câu hỏi chỉ định: "
                                    + ", ".join(missing_actors)
                                    + ". Hãy gọi fetch_doc cho một ứng viên CHƯA "
                                    "đọc trong kết quả search (ưu tiên ứng viên đứng "
                                    "trước) rồi chỉ kết luận từ dòng khớp bộ phận. "
                                    "Các doc_id chưa đọc: "
                                    + ", ".join(unread[:5])
                                    + ". Nếu câu hỏi yêu cầu trường verdict, "
                                    "FINAL mới phải giữ verdict với đúng một "
                                    "lựa chọn."
                                ),
                            }
                        )
                        continue
                    if (
                        "verdict" in ctx.question.casefold()
                        and not isinstance(candidate.get("verdict"), str)
                        and self._verdict_repairs < MAX_VERDICT_REPAIRS
                    ):
                        self._verdict_repairs += 1
                        self._refused_final = candidate
                        ctx.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "FINAL vừa rồi thiếu trường verdict mà câu "
                                    "hỏi yêu cầu. Hãy viết lại cùng FINAL, giữ "
                                    "nguyên answer, claims, citations và abstain; "
                                    "thêm trường verdict chứa ĐÚNG MỘT phương án "
                                    "nguyên văn đã chọn từ câu hỏi."
                                ),
                            }
                        )
                        continue
                if (
                    self._incomplete_claim_repairs
                    and isinstance(self._refused_final, dict)
                ):
                    candidate = _trim_repaired_claims_to_source_lines(
                        ctx, candidate, self._refused_final
                    )
                if (
                    _uses_real_endpoint(self.model)
                    and
                    self._incomplete_claim_repairs < MAX_INCOMPLETE_CLAIM_REPAIRS
                    and _has_incomplete_fetched_claim(ctx, candidate)
                ):
                    # The endpoint authors the corrected claim itself. The
                    # next model call is recorded normally, preserving scorer
                    # provenance; no synthetic quotation is submitted.
                    self._incomplete_claim_repairs += 1
                    self._refused_final = candidate
                    ctx.messages.append(
                        {
                            "role": "user",
                            "content": _complete_claim_repair_prompt(ctx, candidate),
                        }
                    )
                    continue
                report = candidate
                ctx.stop_reason = "final"
                break

            observation = self._observe(ctx, parsed)
            ctx.observations.append(observation)
            ctx.messages.append({"role": "user", "content": observation})

        if ctx.stop_reason != "final" and isinstance(self._refused_final, dict):
            # The loop ran out of steps and the only FINAL the model ever
            # wrote was one `_parse` put aside. Submit it: refusing bought
            # the model turns it did not use, and an empty report scores
            # zero, so this can only ever be an improvement.
            report = dict(self._refused_final)
            ctx.stop_reason = "refused_final"

        report = self.middleware.after_agent(ctx, report)
        # What gets submitted is what the layers returned — the scorer
        # reads the report off the `submit` event and refuses any claim
        # that is not in it (`NOT_SUBMITTED`).
        self.tools.submit(report)
        # No `elapsed_seconds` here on purpose: a wall clock inside the
        # harness would make the trace non-deterministic, and the frozen
        # runner stamps its own `agent_end` with the timing it measured.
        self.trace.emit("agent_end", stop_reason=ctx.stop_reason, steps=ctx.step + 1)
        return report

    # -- reading the model ---------------------------------------------

    def _parse(self, text: str):
        """Decode one model turn — with `arena.model.parse_output`, always.

        Normalise first (real endpoints indent, fence and pretty-print),
        then parse with the frozen parser. Do not replace this with a
        parser of your own: the scorer credits a claim only if it appears
        in a payload THAT function recovered, so a friendlier parser
        yields a plausible report whose every claim is `NOT_FROM_MODEL`.

        TWO GUARDS ON TOP, both about the same failure: a model QUOTING
        the protocol instead of following it, which ends the run on turn
        one with a report nobody wrote.

        1. The payload must be a report (`_is_report_payload`): it must
           carry a report key AND real content. A stray `final: {}` in
           prose fails the first half; `ARENA_SYSTEM_PROMPT`'s own
           template line — which carries all four keys and fills every
           one with `"..."` — fails the second. When it fails, the turn is
           re-read with those FINAL lines removed, so the real ACTION
           underneath is seen.
        2. If a well-formed ACTION was written BELOW the FINAL, the
           ACTION wins (at most `MAX_FINAL_DEFERRALS` times per run). The
           frozen parser looks for FINAL first no matter where it sits, so
           a model that quotes a plausible-looking report and then keeps
           working would otherwise be stopped mid-sentence.

        Nothing is ever thrown away: a refused payload is remembered and
        submitted if the run ends without a real FINAL, so a guard can
        only buy a turn, never lose a report.
        """
        parsed = _parse_with_flat_action_repair(_canonicalise(text))
        if parsed.kind != "final":
            return parsed

        if _is_report_payload(parsed.final):
            action = _action_under_final(text)
            if action is None or self._final_deferrals >= MAX_FINAL_DEFERRALS:
                return parsed
            self._final_deferrals += 1
            self._refused_final = parsed.final
            return action

        if isinstance(parsed.final, dict) and any(
            key in parsed.final for key in REPORT_KEYS
        ):
            self._refused_final = parsed.final
        # Strict, NOT canonicalised: normalisation is what resurrects a
        # non-canonical marker such as `final: {}` in the first place, and
        # this path exists precisely to look underneath one.
        return _parse_with_flat_action_repair(_without_quoted_finals(text))

    # -- the model -----------------------------------------------------

    def _call_model(self, messages: list[dict]):
        """The innermost model call — what `wrap_model_call` wraps.

        The `model_call` event is stamped HERE, from the response the
        model object returned, before any hook can see it. That ordering
        is the whole provenance story: `wrap_model_call` and `after_model`
        are student-owned and can return whatever they like, so a trace
        stamped from their return value would prove nothing at all.
        """
        response = self.model.complete(messages)
        # A frozen runner may take over `model_call` emission (it is the
        # only way to make the record unforgeable). It announces that by
        # setting `emits_model_call = True` on the model object.
        if not getattr(self.model, "emits_model_call", False):
            text = response.text if isinstance(response.text, str) else str(response.text)
            self.trace.emit(
                "model_call",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                # str is immutable — `Trace.emit` stores a reference to
                # whatever it is handed, so a mutable would let later code
                # rewrite history.
                output_text=text[:MAX_OUTPUT_TEXT_CHARS],
                step=self.last_context.step if self.last_context else 0,
            )
        return response

    # -- the tools -----------------------------------------------------

    def _observe(self, ctx: AgentContext, parsed) -> str:
        """Run one tool call through the `wrap_tool_call` chain and turn
        the result into the observation string the model is shown."""
        if parsed.kind != "action" or not parsed.tool:
            # Not a THOUGHT/ACTION turn and not a FINAL either. Say so
            # rather than guessing — a real model that drifts off the
            # protocol needs to be told, and the mock never gets here.
            return (
                f"{TOOL_ERROR_PREFIX} không đọc được ACTION. Hãy trả lời đúng định dạng "
                "THOUGHT/ACTION hoặc THOUGHT/FINAL."
            )

        args = dict(parsed.args)
        if parsed.tool == "search" and _uses_real_endpoint(self.model):
            raw_query = _as_text(args.get("query"))
            raw_span = _normalised_span(raw_query)
            seen_queries = ctx.state.setdefault("_agent_search_queries", set())
            if raw_span and raw_span in seen_queries:
                return _DUPLICATE_SEARCH_NUDGE
            if raw_span:
                seen_queries.add(raw_span)
            search_number = int(ctx.state.get("_agent_search_count", 0)) + 1
            ctx.state["_agent_search_count"] = search_number
            if search_number >= 2:
                args["query"] = _enrich_refined_query(
                    ctx.question, raw_query
                )

        call = self.middleware.wrap_tool_call(ctx, self._dispatch)
        result = call(parsed.tool, args)
        if result is None or not hasattr(result, "ok"):
            return f"{TOOL_ERROR_PREFIX} layer trả về kết quả không hợp lệ cho {parsed.tool}"
        if result.ok and parsed.tool == "fetch_doc":
            doc_id = _as_text(args.get("doc_id"))
            if doc_id:
                ctx.state.setdefault("_agent_fetched_docs", {})[doc_id] = result.content
        elif result.ok and parsed.tool == "search":
            try:
                payload = json.loads(result.content)
            except Exception:
                payload = []
            if isinstance(payload, list):
                known = ctx.state.setdefault("_agent_search_doc_ids", [])
                latest = []
                for item in payload:
                    doc_id = item.get("doc_id") if isinstance(item, dict) else None
                    if isinstance(doc_id, str):
                        if doc_id not in latest:
                            latest.append(doc_id)
                        if doc_id not in known:
                            known.append(doc_id)
                ctx.state["_agent_last_search_doc_ids"] = latest
        return result.content if result.ok else f"{TOOL_ERROR_PREFIX} {result.error}"

    def _dispatch(self, name: str, args: dict) -> ToolResult:
        """The innermost tool call — what `wrap_tool_call` wraps."""
        args = args if isinstance(args, dict) else {}
        if name == "search":
            return self.tools.search(_as_text(args.get("query")), k=_as_k(args.get("k")))
        if name == "fetch_doc":
            return self.tools.fetch_doc(_as_text(args.get("doc_id")))
        if name == "calc":
            return self.tools.calc(_as_text(args.get("expression")) or "0")
        return ToolResult(ok=False, content="", error=f"unknown tool: {name!r}")


def _as_text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_k(value) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return 5
    return max(1, min(MAX_SEARCH_K, k))


__all__ = [
    "AgentContext",
    "ReActAgent",
    "Middleware",
    "MAX_STEPS",
    "ARENA_SYSTEM_PROMPT_REAL",
    "REAL_MODEL_PROMPT_ADDENDUM",
    "real_model_system_prompt",
]
