import os
import re
import zipfile

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ARCHIVE_EXT = (".zip", ".cbz", ".epub")


def is_archive(path):
    return path.lower().endswith(ARCHIVE_EXT)


def _natkey(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_image_entries(zf):
    entries = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        base = os.path.basename(name)
        if not base or base.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXT:
            entries.append(name)
    return sorted(entries, key=_natkey)


def mapping_for(path):
    with zipfile.ZipFile(path) as zf:
        entries = list_image_entries(zf)
    mapping = []
    for idx, entry in enumerate(entries, start=1):
        ext = os.path.splitext(entry)[1].lower()
        mapping.append({"entry": entry, "local": "%05d%s" % (idx, ext)})
    return mapping


def extract(path, workdir):
    os.makedirs(workdir, exist_ok=True)
    mapping = mapping_for(path)
    with zipfile.ZipFile(path) as zf:
        for item in mapping:
            with zf.open(item["entry"]) as src, open(os.path.join(workdir, item["local"]), "wb") as dst:
                dst.write(src.read())
    if not mapping:
        raise ValueError("no image entries found in %s" % path)
    return mapping


def _translated_file(out_dir, local):
    stem = os.path.splitext(local)[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = os.path.join(out_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("translated page missing for %s" % local)


def repack(src_archive, out_dir, mapping, out_archive):
    translated = {item["entry"]: _translated_file(out_dir, item["local"]) for item in mapping}
    out_archive = os.path.abspath(out_archive)
    os.makedirs(os.path.dirname(out_archive), exist_ok=True)
    with zipfile.ZipFile(src_archive) as zin, zipfile.ZipFile(out_archive, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename in translated:
                with open(translated[info.filename], "rb") as handle:
                    zout.writestr(info.filename, handle.read(), compress_type=zipfile.ZIP_DEFLATED)
            else:
                data = zin.read(info.filename)
                compress = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
                zout.writestr(info, data, compress_type=compress)
    return out_archive


def repack_rendered(src_archive, rendered_dir, out_archive):
    return repack(src_archive, rendered_dir, mapping_for(src_archive), out_archive)
