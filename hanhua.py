import argparse
import json
import os
import sys

from mth import pipeline, providers, terms
from mth.config import load_config, find_cjk_font


def load_project(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_project(project, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(project, handle, ensure_ascii=False, indent=2)


def _apply_overrides(cfg, args):
    if getattr(args, "font", None):
        cfg["font"] = args.font
    if getattr(args, "profile", None):
        cfg["profile"] = args.profile
    if getattr(args, "keep_sfx", None) is not None:
        cfg["keep_sfx"] = args.keep_sfx
    if getattr(args, "inpaint", None):
        cfg["inpaint"] = args.inpaint
    if getattr(args, "api_base", None):
        cfg["api_base"] = args.api_base
    if getattr(args, "api_key", None):
        cfg["api_key"] = args.api_key
    if getattr(args, "model", None):
        cfg["text_model"] = args.model
    if getattr(args, "timeout", None):
        cfg["timeout"] = args.timeout
    if getattr(args, "glossary", None):
        cfg["glossary_file"] = args.glossary
    spec = getattr(args, "skip_pages", "")
    if spec:
        ranges = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, _, last = part.partition("-")
                ranges.append([int(first), int(last)])
            else:
                ranges.append([int(part), int(part)])
        cfg["skip_pages"] = ranges
    return cfg


def cmd_extract(args):
    cfg = _apply_overrides(load_config(args.config), args)
    client = providers.build_client(cfg, mock=(args.mock or args.ocr == "mock"))
    if args.ocr == "mock":
        args.mock = True
    project = pipeline.do_extract(args.input, cfg, args.ocr, client)
    out = args.output or os.path.join(args.input, "project.json")
    save_project(project, out)
    blocks = sum(len(p["blocks"]) for p in project["pages"])
    print("extracted %d pages, %d blocks -> %s" % (len(project["pages"]), blocks, out))


def cmd_translate(args):
    cfg = _apply_overrides(load_config(args.config), args)
    client = providers.build_client(cfg, mock=args.mock)
    project = load_project(args.project)
    project = pipeline.do_translate(project, cfg, client)
    out = args.output or args.project
    save_project(project, out)
    done = sum(1 for p in project["pages"] for b in p["blocks"] if b.get("translation"))
    print("translated %d blocks -> %s" % (done, out))


def cmd_render(args):
    cfg = _apply_overrides(load_config(args.config), args)
    find_cjk_font(cfg.get("font") or None)
    project = load_project(args.project)
    out = args.output or os.path.join(os.path.dirname(args.project) or ".", "translated")
    pipeline.do_render(project, args.input, out, cfg)
    print("rendered -> %s" % out)


def cmd_run(args):
    cfg = _apply_overrides(load_config(args.config), args)
    if args.ocr == "mock":
        args.mock = True
    client = providers.build_client(cfg, mock=args.mock)
    project = pipeline.do_extract(args.input, cfg, args.ocr, client)
    project = pipeline.do_translate(project, cfg, client)
    out = args.output or os.path.join(args.input, "translated")
    pipeline.do_render(project, args.input, out, cfg)
    save_project(project, os.path.join(args.input, "project.json"))
    print("done -> %s" % out)


def cmd_unpack(args):
    from mth import archive
    archive.extract(args.input, args.output)
    print("unpacked %d pages -> %s" % (len(archive.mapping_for(args.input)), args.output))


def cmd_repack(args):
    from mth import archive
    archive.repack_rendered(args.input, args.rendered, args.output)
    print("repacked -> %s" % args.output)


def cmd_terms(args):
    cfg = _apply_overrides(load_config(args.config), args)
    client = providers.build_client(cfg, mock=args.mock)
    project = load_project(args.project)
    candidates = terms.extract_candidates(project, min_count=args.min_count, top_k=args.top_k)
    contexts = terms.contexts_for(project, candidates)
    proposed = terms.propose_glossary(client, cfg["text_model"], candidates, contexts, cfg["target_lang"])
    out = args.output or (os.path.splitext(args.project)[0] + ".glossary.json")
    terms.write_glossary(out, proposed, cfg["target_lang"])
    print("candidates: %d, kept: %d -> %s" % (len(candidates), len(proposed), out))
    print("請人工/AI 校對 %s（改 zh 欄），完成後再 translate --glossary" % out)


def cmd_normalize(args):
    cfg = _apply_overrides(load_config(args.config), args)
    if not args.glossary:
        raise SystemExit("normalize requires --glossary FILE")
    client = providers.build_client(cfg, mock=args.mock)
    project = load_project(args.project)
    _, flat_map = terms.load_glossary(args.glossary)
    fixed = terms.enforce_glossary(project, flat_map, client, cfg)
    out = args.output or args.project
    save_project(project, out)
    print("normalized %d blocks -> %s" % (fixed, out))


def cmd_verify(args):
    _apply_overrides(load_config(args.config), args)
    if not args.glossary:
        raise SystemExit("verify requires --glossary FILE")
    project = load_project(args.project)
    _, flat_map = terms.load_glossary(args.glossary)
    violations, variants = terms.verify_glossary(project, flat_map)
    for src, counter in variants.items():
        if len(counter) > 1 or "<<不一致>>" in counter:
            print("譯名不一致: %s -> %s" % (src, dict(counter)))
    if violations:
        print("INCONSISTENT: %d blocks" % len(violations))
        for item in violations[:20]:
            print("  %s: 應含「%s」 目前「%s」" % (item["id"], item["expected"], item["translation"]))
    else:
        print("OK: glossary consistent (%d terms)" % len(flat_map))
    if violations and args.fail:
        raise SystemExit(1)


def default_output(path, suffix="_MTL_zhTW"):
    stem, ext = os.path.splitext(path)
    return stem + suffix + ext


def _all_katakana(text):
    return bool(text) and all("\u30a0" <= ch <= "\u30ff" for ch in text)


def _auto_glossary(project, cfg, client, path):
    from mth import terms
    candidates = terms.extract_candidates(project, min_count=2, top_k=60)
    pairs = [{"id": str(i), "ja": c["src"]} for i, c in enumerate(candidates)]
    mapping = providers.translate_batch(client, cfg["text_model"], pairs, None,
                                        cfg["target_lang"], cfg.get("profile", "sakura"))
    keep = []
    for i, c in enumerate(candidates):
        zh = (mapping.get(str(i)) or "").strip()
        src = c["src"]
        if not zh:
            continue
        if any(s in src for s in ("君", "さん", "ちゃん", "先輩", "様", "氏")) or _all_katakana(src):
            keep.append({"src": src, "zh": zh, "type": "name",
                         "count": c["count"], "status": "auto"})
    terms.write_glossary(path, keep, cfg["target_lang"])
    print("auto-terms: kept %d names" % len(keep))
    return path


def cmd_auto(args):
    import shutil
    import tempfile
    from mth import archive, terms
    cfg = _apply_overrides(load_config(args.config), args)
    cfg["keep_ext"] = True
    if args.ocr == "mock":
        args.mock = True
    client = providers.build_client(cfg, mock=args.mock)
    if not archive.is_archive(args.input):
        raise SystemExit("auto expects .zip/.cbz/.epub: %s" % args.input)
    out = args.output or default_output(args.input)
    work = tempfile.mkdtemp(prefix="hanhua_")
    raw = os.path.join(work, "raw")
    rendered = os.path.join(work, "out")
    ok = False
    try:
        mapping = archive.extract(args.input, raw)
        if cfg.get("unload_llm_before_ocr"):
            if providers.unload_model(cfg):
                print("ocr: released the translation model's VRAM for inpainting")
        project = pipeline.do_extract(raw, cfg, args.ocr, client)
        paired = pipeline.pair_by_names(project, mapping)
        if paired:
            print("clean twin: %d page(s) will be typeset onto the textless copy "
                  "(no inpainting)" % paired)
        gpath = args.glossary or cfg.get("glossary_file") or ""
        if not gpath and args.auto_terms:
            gpath = _auto_glossary(project, cfg, client, os.path.join(work, "glossary.json"))
        if gpath:
            cfg["glossary_file"] = gpath
        project = pipeline.do_translate(project, cfg, client)
        if gpath:
            _, flat = terms.load_glossary(gpath)
            fixed = terms.enforce_glossary(project, flat, client, cfg)
            violations, _ = terms.verify_glossary(project, flat)
            print("normalize fixed %d; glossary violations %d" % (fixed, len(violations)))
            for item in violations[:10]:
                print("  ! %s src=%s expected=%s" % (item["id"], item["src"], item["expected"]))
            if violations and getattr(args, "strict", False):
                save_project(project, out + ".project.json")
                raise SystemExit("glossary inconsistent: %d blocks (--strict)" % len(violations))
        pipeline.do_render(project, raw, rendered, cfg)
        archive.repack(args.input, rendered, mapping, out)
        save_project(project, out + ".project.json")
        ok = True
    finally:
        # Keep the work dir on failure (it holds the LaMa-cleaned pages) so the
        # run can be inspected/resumed instead of losing everything.
        if ok and not getattr(args, "keep_work", False):
            shutil.rmtree(work, ignore_errors=True)
        else:
            print("work dir kept for inspection: %s" % work)
    print("MTL done -> %s" % out)
    print("project  -> %s" % (out + ".project.json"))


def cmd_vast(args):
    from mth import vast
    key = args.api_key or os.environ.get("VAST_API_KEY", "")
    if not key:
        raise SystemExit("set VAST_API_KEY or pass --api-key")
    deploy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy")
    workdir = os.path.dirname(os.path.abspath(__file__))

    if args.action == "status":
        acct = vast.account(key)
        print("credit: %s | total_spend: %s" % (acct.get("credit"), acct.get("total_spend")))
        inst = vast.list_instances(key)
        print("instances: %d" % len(inst))
        for i in inst:
            print("  - %s %s %s" % (i.get("id"), i.get("actual_status"), i.get("gpu_name")))
        return

    if args.action == "search":
        for o in vast.search_offers(key, min_down=args.min_down, min_up=args.min_up,
                                    max_price=args.max_price, gpu=args.gpu):
            print("%-9s %-12s %4dG $%.3f down=%.0f rel=%.3f" % (
                o.get("id"), o.get("gpu_name"), int(o.get("gpu_ram", 0) / 1024),
                o.get("dph_total", 0.0), o.get("inet_down", 0.0), o.get("reliability2", 0.0)))
        return

    if args.action == "down":
        ids = vast.destroy_all(key)
        print("destroyed: %s | remaining: %d" % (ids, len(vast.list_instances(key))))
        return

    endpoints = vast.up(key, workdir=workdir, deploy_dir=deploy_dir,
                        model=args.model or vast.DEFAULT_MODEL,
                        min_down=args.min_down, min_up=args.min_up,
                        max_price=args.max_price, gpu=args.gpu, disk=args.disk,
                        image=args.image or vast.DEFAULT_IMAGE, baked=args.baked,
                        dry_run=args.dry_run, yes=args.yes, attempts=args.attempts)
    if not endpoints:
        print("dry-run only; add --yes to actually rent")
        return
    print(json.dumps(endpoints, ensure_ascii=False, indent=2))
    if args.write_config:
        cfg = {}
        if os.path.exists(args.write_config):
            with open(args.write_config, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        cfg.update({"profile": "sakura", "api_base": endpoints["api_base"], "api_key": "ollama",
                    "text_model": endpoints["text_model"], "ocr_url": endpoints["ocr_url"]})
        with open(args.write_config, "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, ensure_ascii=False, indent=2)
        print("wrote endpoints -> %s" % args.write_config)


def cmd_book(args):
    import shutil
    import tempfile
    from mth import archive
    cfg = _apply_overrides(load_config(args.config), args)
    cfg["keep_ext"] = True
    if args.ocr == "mock":
        args.mock = True
    client = providers.build_client(cfg, mock=args.mock)
    if not archive.is_archive(args.input):
        raise ValueError("book mode expects a .zip/.cbz/.epub: %s" % args.input)
    dest = args.output or default_output(args.input)
    workdir = tempfile.mkdtemp(prefix="hanhua_")
    raw = os.path.join(workdir, "raw")
    out = os.path.join(workdir, "out")
    try:
        mapping = archive.extract(args.input, raw)
        if cfg.get("unload_llm_before_ocr"):
            if providers.unload_model(cfg):
                print("ocr: released the translation model's VRAM for inpainting")
        project = pipeline.do_extract(raw, cfg, args.ocr, client)
        paired = pipeline.pair_by_names(project, mapping)
        if paired:
            print("clean twin: %d page(s) will be typeset onto the textless copy "
                  "(no inpainting)" % paired)
        project = pipeline.do_translate(project, cfg, client)
        pipeline.do_render(project, raw, out, cfg)
        archive.repack(args.input, out, mapping, dest)
        save_project(project, dest + ".project.json")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    pages = len(mapping)
    print("book done: %d pages -> %s" % (pages, dest))
    print("project sidecar -> %s" % (dest + ".project.json"))


def cmd_ping(args):
    cfg = _apply_overrides(load_config(args.config), args)
    client = providers.build_client(cfg, mock=args.mock)
    print("endpoint : %s" % cfg["api_base"])
    print("model    : %s" % cfg["text_model"])
    print("profile  : %s" % cfg.get("profile", "openai"))
    reply = providers.ping(client, cfg)
    print("reply    : %s" % reply)


def cmd_selftest(args):
    from PIL import Image, ImageDraw
    base = os.path.abspath(args.output or os.path.join("selftest_out"))
    raw = os.path.join(base, "raw")
    out = os.path.join(base, "translated")
    os.makedirs(raw, exist_ok=True)
    img = Image.new("RGB", (900, 1300), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    box = (int(900 * 0.12), int(1300 * 0.08), int(900 * 0.88), int(1300 * 0.22))
    draw.rectangle(box, fill=(255, 255, 255), outline=(60, 60, 60), width=3)
    for i in range(6):
        x = box[0] + 40 + i * 90
        draw.line([(x, box[1] + 30), (x, box[3] - 30)], fill=(40, 40, 40), width=8)
    page_path = os.path.join(raw, "001.png")
    img.save(page_path)
    cfg = load_config()
    cfg.update({"inpaint": "fill", "keep_sfx": True, "profile": args.profile})
    client = providers.build_client(cfg, mock=True)
    project = pipeline.do_extract(raw, cfg, "mock", client)
    project = pipeline.do_translate(project, cfg, client)
    pipeline.do_render(project, raw, out, cfg)
    save_project(project, os.path.join(base, "project.json"))
    result = os.path.join(out, "001.png")
    print("selftest ok")
    print("  input  : %s" % page_path)
    print("  output : %s" % result)
    print("  project: %s" % os.path.join(base, "project.json"))


def build_parser():
    parser = argparse.ArgumentParser(prog="hanhua", description="OCR-first JSON round-trip manga translator")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--config", help="path to config.json")
        p.add_argument("--mock", action="store_true", help="use offline mock provider")
        p.add_argument("--font", help="CJK font path")
        p.add_argument("--profile", choices=["openai", "sakura"], help="translation prompt profile")
        p.add_argument("--model", help="override text model name")
        p.add_argument("--timeout", type=int, help="HTTP timeout seconds")
        p.add_argument("--glossary", help="glossary.json to enforce")
        p.add_argument("--api-base")
        p.add_argument("--api-key")

    e = sub.add_parser("extract", help="detect + OCR into project JSON")
    e.add_argument("-i", "--input", required=True)
    e.add_argument("-o", "--output")
    e.add_argument("--ocr", default="vision", choices=["vision", "opencv", "sidecar", "http", "manga_ocr", "mock"])
    add_common(e)
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("translate", help="translate project JSON via LLM")
    t.add_argument("-p", "--project", required=True)
    t.add_argument("-o", "--output")
    add_common(t)
    t.set_defaults(func=cmd_translate)

    r = sub.add_parser("render", help="erase + typeset back into images")
    r.add_argument("-p", "--project", required=True)
    r.add_argument("-i", "--input", required=True)
    r.add_argument("-o", "--output")
    r.add_argument("--inpaint", choices=["fill", "blur", "opencv", "lama"])
    r.add_argument("--keep-sfx", dest="keep_sfx", action="store_true", default=None)
    r.add_argument("--no-keep-sfx", dest="keep_sfx", action="store_false")
    add_common(r)
    r.set_defaults(func=cmd_render)

    b = sub.add_parser("book", help="translate a .zip/.cbz/.epub and repack it")
    b.add_argument("-i", "--input", required=True, help="source archive")
    b.add_argument("-o", "--output", help="output archive (default <name>_MTL_zhTW<ext>)")
    b.add_argument("--ocr", default="vision", choices=["vision", "opencv", "sidecar", "http", "manga_ocr", "mock"])
    b.add_argument("--inpaint", choices=["fill", "blur", "opencv", "lama"])
    add_common(b)
    b.set_defaults(func=cmd_book)

    a = sub.add_parser("auto", help="one-shot: archive in -> <name>_MTL_zhTW archive out (tool does everything)")
    a.add_argument("-i", "--input", required=True, help="source archive")
    a.add_argument("-o", "--output", help="output archive (default <name>_MTL_zhTW<ext>)")
    a.add_argument("--ocr", default="http", choices=["vision", "opencv", "sidecar", "http", "manga_ocr", "mock"])
    a.add_argument("--inpaint", choices=["fill", "blur", "opencv", "lama"])
    a.add_argument("--auto-terms", dest="auto_terms", action="store_true",
                   help="auto-build a best-effort name glossary")
    a.add_argument("--strict", action="store_true",
                   help="abort if the glossary is inconsistent (default: warn and continue)")
    a.add_argument("--keep-work", dest="keep_work", action="store_true",
                   help="keep the temp work dir even on success")
    a.add_argument("--skip-pages", dest="skip_pages", default="",
                   help="page ranges to leave untouched (textless copies), e.g. 13-24,40-52")
    add_common(a)
    a.set_defaults(func=cmd_auto)

    v = sub.add_parser("vast", help="vast.ai lifecycle: status / search / up / down")
    v.add_argument("action", choices=["status", "search", "up", "down"])
    v.add_argument("--api-key")
    v.add_argument("--gpu")
    v.add_argument("--min-down", type=int, default=300)
    v.add_argument("--min-up", type=int, default=100)
    v.add_argument("--max-price", type=float, default=0.18)
    v.add_argument("--model")
    v.add_argument("--disk", type=int, default=60)
    v.add_argument("--image", help="docker image to boot (default: pytorch base image)")
    v.add_argument("--baked", action="store_true",
                   help="image already contains the whole stack; skip onstart install")
    v.add_argument("--attempts", type=int, default=3, help="try up to N machines if NET_BAD")
    v.add_argument("--dry-run", dest="dry_run", action="store_true")
    v.add_argument("--yes", action="store_true")
    v.add_argument("--write-config", dest="write_config",
                   help="write api_base/ocr_url into this config.json")
    v.set_defaults(func=cmd_vast)

    u = sub.add_parser("run", help="extract + translate + render")
    u.add_argument("-i", "--input", required=True)
    u.add_argument("-o", "--output")
    u.add_argument("--ocr", default="vision", choices=["vision", "opencv", "sidecar", "http", "manga_ocr", "mock"])
    add_common(u)
    u.set_defaults(func=cmd_run)

    s = sub.add_parser("selftest", help="offline end-to-end test with mock provider")
    s.add_argument("-o", "--output")
    s.add_argument("--profile", choices=["openai", "sakura"], default="openai")
    s.set_defaults(func=cmd_selftest)

    p = sub.add_parser("ping", help="test the translation endpoint")
    add_common(p)
    p.set_defaults(func=cmd_ping)

    up = sub.add_parser("unpack", help="unpack an archive's pages to a folder")
    up.add_argument("-i", "--input", required=True)
    up.add_argument("-o", "--output", required=True)
    up.set_defaults(func=cmd_unpack)

    rp = sub.add_parser("repack", help="rebuild an archive from a rendered folder")
    rp.add_argument("-i", "--input", required=True, help="original archive")
    rp.add_argument("--rendered", required=True, help="folder of rendered pages")
    rp.add_argument("-o", "--output", required=True, help="output archive")
    rp.set_defaults(func=cmd_repack)

    tm = sub.add_parser("terms", help="extract + propose glossary for agent review")
    tm.add_argument("-p", "--project", required=True)
    tm.add_argument("-o", "--output")
    tm.add_argument("--min-count", type=int, default=2)
    tm.add_argument("--top-k", type=int, default=300)
    add_common(tm)
    tm.set_defaults(func=cmd_terms)

    nm = sub.add_parser("normalize", help="enforce glossary consistency in translations")
    nm.add_argument("-p", "--project", required=True)
    nm.add_argument("-o", "--output")
    add_common(nm)
    nm.set_defaults(func=cmd_normalize)

    vf = sub.add_parser("verify", help="verify glossary consistency (exit 1 on violation)")
    vf.add_argument("-p", "--project", required=True)
    vf.add_argument("--no-fail", dest="fail", action="store_false", default=True)
    add_common(vf)
    vf.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
