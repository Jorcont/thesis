from pathlib import Path


def template_to_path(path: str | Path) -> Path:
    path = str(path)

    project_root = Path(__file__).resolve().parent.parent
    output_root = project_root / "output"
    path = path.replace("<project>", str(project_root))
    path = path.replace("<output>", str(output_root))
    return Path(path)

def template_to_pathstr(path: str | Path) -> str:
    return str(template_to_path(path))