import sys
from pathlib import Path
import xml.etree.ElementTree as ET

FOLDER_LOCATION = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"

def build_xml(dir_path: Path, root_level: bool = True) -> ET.Element:
    """
    Recursively build the XML tree.
    Root: <dataset>
    Subfolders: <folder name="...">
    Files: <file name="...">
    """
    if root_level:
        elem = ET.Element("dataset")
    else:
        elem = ET.Element("folder", {"name": dir_path.name})

    # Add files
    for file in sorted(dir_path.iterdir()):
        if file.is_file():
            elem.append(ET.Element("file", {"name": file.name}))

    # Add subdirectories
    for sub in sorted(dir_path.iterdir()):
        if sub.is_dir():
            elem.append(build_xml(sub, root_level=False))

    return elem

def indent(elem: ET.Element, level: int = 0) -> None:
    """
    In-place pretty printer (ElementTree doesn't indent by default).
    """
    i = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():  # last child's tail
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def main(directory: str = FOLDER_LOCATION) -> None:
    path = Path(directory)
    if not path.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    root_elem = build_xml(path, root_level=True)
    indent(root_elem)

    tree = ET.ElementTree(root_elem)
    # Write to stdout with XML declaration
    xml_bytes = ET.tostring(root_elem, encoding="utf-8")
    xml_string = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
    print(xml_string.decode("utf-8"))

if __name__ == "__main__":
    # Optional: allow passing a custom path
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()