import sys
sys.path.insert(0, ".")
from tools.document_parser import parse_document

print("Starting PDF parsing...")
chunks = parse_document(r"tests\test_sample.pdf")
print(f"Parsed {len(chunks)} chunks:")
for c in chunks:
    typ = c.metadata.get("type", "?")
    print(f"  [{c.chunk_index}] page={c.page_number} type={typ}")
    print(f"      {c.content[:120]}")
print("DONE")
