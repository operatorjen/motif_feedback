from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .atomic_files import atomic_write_bytes
from .constants import (
    PROJECT_FILE_LIST_MAX_ENTRIES,
    PROJECT_FILE_READ_MAX_CHARS,
    PROJECT_FILE_SEARCH_DEFAULT_RESULTS,
    PROJECT_FILE_SEARCH_MAX_RESULTS,
    PROJECT_FILE_SEARCH_SCAN_MAX_CHARS,
    PROJECT_FILE_SEARCH_SNIPPET_AFTER_CHARS,
    PROJECT_FILE_SEARCH_SNIPPET_BEFORE_CHARS,
)
from .storage import Storage, StorageError

TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".html",
    ".css",
    ".js",
    ".py",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".toml",
    ".ini",
    ".xml",
    ".sql",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".svg",
}
RASTER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | RASTER_IMAGE_SUFFIXES
IGNORED_RUNTIME_DIRECTORY_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
IGNORED_RUNTIME_FILE_NAMES = {".DS_Store"}
IGNORED_RUNTIME_SUFFIXES = {".pyc", ".pyo"}
CODE_SUFFIXES = {
    ".json", ".yaml", ".yml", ".csv", ".html", ".css", ".js", ".py",
    ".ts", ".tsx", ".jsx", ".sh", ".toml", ".ini", ".xml", ".sql",
    ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs",
}

TEXT_DECODING_ERRORS = "replace"


class FileToolError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "file_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProjectFileTools:
    def __init__(self, storage: Storage, max_write_bytes: int, max_upload_bytes: int) -> None:
        self.storage = storage
        self.projects_root = storage.projects_root.resolve()
        self.max_write_bytes = max_write_bytes
        self.max_upload_bytes = max_upload_bytes

    def project_root(self, project_id: str) -> Path:
        identifier = self.storage.validate_project_id(project_id)
        self.storage.get_project(identifier)
        root = (self.projects_root / identifier).resolve(strict=False)
        root.relative_to(self.projects_root)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise FileToolError("Project directories may not be symbolic links.")
        return root

    def confined_path(self, project_id: str, relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise FileToolError("A valid relative path is required.")
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise FileToolError("Absolute paths are not allowed.")
        if any(part in {"", ".", ".."} for part in supplied.parts):
            raise FileToolError("Path traversal and empty path components are not allowed.")

        root = self.project_root(project_id)
        current = root
        for part in supplied.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise FileToolError("Symbolic links are not allowed.")

        candidate = (root / supplied).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FileToolError("Path is outside the project workspace.") from exc
        return candidate

    @staticmethod
    def _validate_suffix(path: Path) -> None:
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
            raise FileToolError(f"File type is not allowed. Allowed extensions: {allowed}")

    @staticmethod
    def _file_kind(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in RASTER_IMAGE_SUFFIXES or suffix == ".svg":
            return "image"
        if suffix in CODE_SUFFIXES:
            return "code"
        if suffix in {".md", ".markdown"}:
            return "markdown"
        return "text"

    @staticmethod
    def _is_runtime_artifact(path: Path) -> bool:
        return (
            any(part in IGNORED_RUNTIME_DIRECTORY_NAMES for part in path.parts)
            or path.name in IGNORED_RUNTIME_FILE_NAMES
            or path.suffix.lower() in IGNORED_RUNTIME_SUFFIXES
        )

    def list_files(self, project_id: str) -> list[dict]:
        root = self.project_root(project_id)
        owners = self.storage.file_owners(project_id)
        output: list[dict] = []
        for path in sorted(root.rglob("*")):
            if len(output) >= PROJECT_FILE_LIST_MAX_ENTRIES:
                break
            if path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if self._is_runtime_artifact(relative):
                continue
            stat = path.stat()
            normalized = relative.as_posix()
            output.append(
                {
                    "path": normalized,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "kind": self._file_kind(path),
                    **owners.get(normalized, {}),
                }
            )
        return output

    def read_file(
        self,
        project_id: str,
        relative_path: str,
        max_chars: int = PROJECT_FILE_READ_MAX_CHARS,
    ) -> dict:
        path = self.confined_path(project_id, relative_path)
        self._validate_suffix(path)
        if not path.exists() or not path.is_file():
            raise FileToolError("File not found.")
        if path.suffix.lower() in RASTER_IMAGE_SUFFIXES:
            raise FileToolError(
                "Binary images can be opened in Preview but not read as text.",
                code="binary_preview_only",
            )
        stat = path.stat()
        if stat.st_size > self.max_upload_bytes:
            raise FileToolError("File is too large to read through this interface.")
        with path.open(
            "r",
            encoding="utf-8",
            errors=TEXT_DECODING_ERRORS,
        ) as handle:
            content = handle.read(max_chars + 1)
        truncated = len(content) > max_chars
        return {
            "path": relative_path,
            "content": content[:max_chars],
            "truncated": truncated,
            "size_bytes": stat.st_size,
        }

    def write_file(
        self,
        project_id: str,
        relative_path: str,
        content: str,
        *,
        actor_type: str = "user",
        actor_id: str | None = None,
    ) -> dict:
        path = self.confined_path(project_id, relative_path)
        self._validate_suffix(path)
        if path.suffix.lower() in RASTER_IMAGE_SUFFIXES:
            raise FileToolError(
                "Agents cannot encode raster-image bytes through the text file tool. "
                "Create a safe .svg graphic instead.",
                code="binary_write_not_allowed",
            )
        if path.suffix.lower() == ".svg":
            self._validate_svg(content)
        encoded = content.encode("utf-8")
        if actor_type == "agent" and len(encoded) > self.max_write_bytes:
            raise FileToolError(
                f"This agent-owned file would be {len(encoded)} bytes, above the "
                f"{self.max_write_bytes}-byte maximum. Reread the existing file, reframe its "
                "contents, remove repetition, and replace it with a consolidated version that "
                "preserves the durable observations within the limit.",
                code="agent_file_size_limit",
                retryable=True,
            )
        if actor_type != "agent" and len(encoded) > self.max_upload_bytes:
            raise FileToolError(
                "Upload exceeds the configured size limit.",
                code="upload_size_limit",
            )
        existed = path.exists()
        project_root = self.project_root(project_id)
        normalized_path = path.relative_to(project_root).as_posix()
        owner: dict | None = None
        if existed:
            owner = self.storage.get_file_owner(project_id, normalized_path)
            if actor_type != "agent" or not actor_id:
                raise FileToolError("File already exists.")
            permitted = bool(
                owner
                and owner.get("owner_type") == "agent"
                and (
                    owner.get("owner_id") == actor_id
                    or owner.get("shared_agent_edit")
                )
            )
            if not permitted:
                raise FileToolError(
                    "You may edit files you created, or an agent-created file the user explicitly shared."
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in [path.parent, *path.parents]:
            if parent == project_root.parent:
                break
            if parent.exists() and parent.is_symlink():
                raise FileToolError("Symbolic links are not allowed.")

        atomic_write_bytes(path, encoded)
        if existed:
            self.storage.touch_file_owner(project_id, normalized_path)
        else:
            self.storage.record_file_owner(project_id, normalized_path, actor_type, actor_id)
        final_owner_type = owner.get("owner_type") if owner else actor_type
        final_owner_id = owner.get("owner_id") if owner else actor_id
        return {
            "ok": True,
            "path": normalized_path,
            "bytes_written": len(encoded),
            "overwritten": existed,
            "owner_type": final_owner_type,
            "owner_id": final_owner_id,
            "shared_agent_edit": bool(owner and owner.get("shared_agent_edit")),
        }

    def set_agent_sharing(
        self, project_id: str, relative_path: str, allowed: bool
    ) -> dict:
        path = self.confined_path(project_id, relative_path)
        self._validate_suffix(path)
        if not path.exists() or not path.is_file():
            raise FileToolError("File not found.")
        normalized_path = path.relative_to(self.project_root(project_id)).as_posix()
        try:
            return self.storage.set_file_sharing(project_id, normalized_path, allowed)
        except StorageError as exc:
            raise FileToolError(str(exc), code="file_sharing_not_allowed") from exc

    def save_upload(self, project_id: str, filename: str, content: bytes) -> dict:
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name:
            raise FileToolError("Upload filename is invalid.")
        if len(content) > self.max_upload_bytes:
            raise FileToolError("Upload exceeds the configured size limit.")
        path = self.confined_path(project_id, safe_name)
        self._validate_suffix(path)
        if path.suffix.lower() in RASTER_IMAGE_SUFFIXES:
            self._validate_raster_signature(path.suffix.lower(), content)
            if path.exists():
                raise FileToolError("File already exists.")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, content)
            normalized_path = path.relative_to(self.project_root(project_id)).as_posix()
            self.storage.record_file_owner(project_id, normalized_path, "user", None)
            return {
                "ok": True,
                "path": normalized_path,
                "bytes_written": len(content),
                "overwritten": False,
                "owner_type": "user",
                "owner_id": None,
                "kind": "image",
            }
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FileToolError("Only UTF-8 text files are accepted in this starter.") from exc
        return self.write_file(project_id, safe_name, text, actor_type="user")

    def preview_path(self, project_id: str, relative_path: str) -> tuple[Path, str]:
        path = self.confined_path(project_id, relative_path)
        self._validate_suffix(path)
        if not path.exists() or not path.is_file():
            raise FileToolError("File not found.")
        suffix = path.suffix.lower()
        media_types = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        if suffix not in media_types:
            raise FileToolError("This file does not have an image preview.")
        if suffix == ".svg":
            try:
                svg_text = path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise FileToolError("SVG previews must be valid UTF-8 text.") from exc
            self._validate_svg(svg_text)
        return path, media_types[suffix]

    def download_path(self, project_id: str, relative_path: str) -> Path:
        path = self.confined_path(project_id, relative_path)
        self._validate_suffix(path)
        if not path.exists() or not path.is_file():
            raise FileToolError("File not found.")
        return path

    @staticmethod
    def _validate_raster_signature(suffix: str, content: bytes) -> None:
        valid = {
            ".png": content.startswith(b"\x89PNG\r\n\x1a\n"),
            ".jpg": content.startswith(b"\xff\xd8\xff"),
            ".jpeg": content.startswith(b"\xff\xd8\xff"),
            ".gif": content.startswith((b"GIF87a", b"GIF89a")),
            ".webp": len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP",
        }
        if not valid.get(suffix, False):
            raise FileToolError("The uploaded image contents do not match its file extension.")

    @staticmethod
    def _validate_svg(content: str) -> None:
        lowered = content.lower()
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise FileToolError("SVG documents may not contain document types or entities.")
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise FileToolError(f"SVG is not valid XML: {exc}") from exc
        if root.tag.rsplit("}", 1)[-1].lower() != "svg":
            raise FileToolError("An SVG graphic must have an <svg> root element.")
        forbidden_elements = {
            "script", "foreignobject", "iframe", "object", "embed", "audio",
            "video", "image", "style", "a",
        }
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""
            if tag in forbidden_elements:
                raise FileToolError(f"SVG element <{tag}> is not allowed in project previews.")
            for raw_name, raw_value in element.attrib.items():
                name = raw_name.rsplit("}", 1)[-1].lower()
                value = str(raw_value).strip().lower()
                if name.startswith("on"):
                    raise FileToolError("SVG event-handler attributes are not allowed.")
                if name in {"href", "src"} and not value.startswith("#"):
                    raise FileToolError("SVG references must remain inside the same graphic.")
                if any(marker in value for marker in ("javascript:", "data:", "http:", "https:", "//")):
                    raise FileToolError("SVG previews may not reference scripts, data URLs, or networks.")
                if re.search(r"url\(\s*(?!#)", value):
                    raise FileToolError("SVG URL references must point to a local element ID.")

    def delete_file(self, project_id: str, relative_path: str) -> dict:
        path = self.confined_path(project_id, relative_path)
        if not path.exists() or not path.is_file():
            raise FileToolError("File not found.")
        root = self.project_root(project_id)
        normalized_path = path.relative_to(root).as_posix()
        parent = path.parent
        path.unlink()
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        self.storage.remove_file_owner(project_id, normalized_path)
        return {"ok": True, "path": normalized_path, "deleted": True}

    def search_files(
        self,
        project_id: str,
        query: str,
        max_results: int = PROJECT_FILE_SEARCH_DEFAULT_RESULTS,
    ) -> list[dict]:
        clean_query = " ".join(query.split()).strip()
        if not clean_query:
            raise FileToolError("Search query is required.")
        terms = [term.lower() for term in re.findall(r"[\w-]+", clean_query) if len(term) > 1]
        if not terms:
            terms = [clean_query.lower()]

        result_limit = min(
            max(max_results, 1),
            PROJECT_FILE_SEARCH_MAX_RESULTS,
        )
        results: list[dict] = []
        for file_info in self.list_files(project_id):
            path = file_info["path"]
            try:
                data = self.read_file(
                    project_id,
                    path,
                    max_chars=PROJECT_FILE_SEARCH_SCAN_MAX_CHARS,
                )
            except FileToolError:
                continue
            text = data["content"]
            lowered = text.lower()
            score = sum(lowered.count(term) for term in terms)
            if score <= 0:
                continue
            first_position = min((lowered.find(term) for term in terms if term in lowered), default=0)
            start = max(
                0,
                first_position - PROJECT_FILE_SEARCH_SNIPPET_BEFORE_CHARS,
            )
            end = min(
                len(text),
                first_position + PROJECT_FILE_SEARCH_SNIPPET_AFTER_CHARS,
            )
            snippet = text[start:end].replace("\n", " ").strip()
            results.append({"path": path, "score": score, "snippet": snippet})

        results.sort(key=lambda item: (-item["score"], item["path"]))
        return results[:result_limit]
