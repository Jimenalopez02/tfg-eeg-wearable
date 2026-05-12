from pathlib import Path
from collections import Counter

RAW_ROOTS = [
    Path(r"..\SEED_VIG\Raw_Data").resolve(),
    Path(r"..\SEED_VIG").resolve(),
]

print("=== Buscando datos raw de somnolencia ===")

existing_roots = [p for p in RAW_ROOTS if p.exists()]
if not existing_roots:
    print("No encuentro ninguna de estas rutas:")
    for p in RAW_ROOTS:
        print(" -", p)
    raise SystemExit

for root in existing_roots:
    print(f"\n[ROOT] {root}")

    files = [p for p in root.rglob("*") if p.is_file()]
    print("Nº total de archivos:", len(files))

    ext_counter = Counter(p.suffix.lower() for p in files)
    print("Extensiones encontradas:")
    for ext, n in ext_counter.most_common():
        print(f"  {ext or '[sin extensión]'}: {n}")

    print("\nPrimeros 30 archivos:")
    for p in files[:30]:
        print(" -", p.relative_to(root))