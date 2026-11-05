"""import_offer / import_offer_apply (bigbuild 2026-10-28: 94 of 109 refused writes were
"undefined name" refusals for names a sibling module defined; the model rewrote that module
unchanged instead of adding the import)."""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off

from vega import features
from vega.tools import execute

USE = "def f(x):\n    if not x:\n        raise ValidationError('no')\n    return x\n"


def project(tmp_path: Path, service: str = "from .models import X\n\n\ndef f(x):\n    return x\n"
            ) -> Path:
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger/__init__.py").write_text("")
    (tmp_path / "ledger/errors.py").write_text("class ValidationError(Exception):\n    pass\n")
    (tmp_path / "ledger/models.py").write_text("X = 1\n")
    (tmp_path / "ledger/service.py").write_text(service)
    return tmp_path


def replace(tmp_path: Path, code: str = USE) -> str:
    return execute("replace_definition", {"path": "ledger/service.py", "name": "f", "code": code},
                   tmp_path)


def test_the_refusal_names_the_module_and_the_import(tmp_path: Path) -> None:
    result = replace(project(tmp_path))
    assert result.startswith("Write not applied")
    assert ("`ValidationError` is defined in ledger/errors.py: add "
            "`from .errors import ValidationError` to the imports at the top of "
            "ledger/service.py (edit_file), then send this write again") in result
    assert "for the other undefined names" not in result
    assert "ValidationError" not in (tmp_path / "ledger/service.py").read_text()  # not applied


def test_the_offer_follows_the_files_absolute_import_style(tmp_path: Path) -> None:
    project(tmp_path, "from ledger.models import X\n\n\ndef f(x):\n    return x\n")
    assert "`from ledger.errors import ValidationError`" in replace(tmp_path)


def test_a_whole_file_write_is_told_to_add_it_in_the_same_write(tmp_path: Path) -> None:
    project(tmp_path)
    result = execute("write_file", {"path": "ledger/service.py", "content": USE}, tmp_path)
    assert "add `from .errors import ValidationError` at the top of the file in the same " \
           "write" in result


def test_no_offer_for_a_name_defined_nowhere(tmp_path: Path) -> None:
    result = replace(project(tmp_path), USE.replace("ValidationError", "Missing"))
    assert "is defined in" not in result
    assert "define or import each undefined name" in result


def test_no_offer_when_two_modules_define_it(tmp_path: Path) -> None:
    project(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other/errors.py").write_text("class ValidationError(Exception):\n    pass\n")
    (tmp_path / "other/more.py").write_text("ValidationError = ValueError\n")
    (tmp_path / "ledger/more.py").write_text("class ValidationError(Exception):\n    pass\n")
    assert "is defined in" not in replace(tmp_path)


def test_two_homes_but_one_in_the_files_own_package_is_offered(tmp_path: Path) -> None:
    project(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other/errors.py").write_text("class ValidationError(Exception):\n    pass\n")
    assert "`ValidationError` is defined in ledger/errors.py" in replace(tmp_path)


def test_no_offer_when_the_import_would_be_circular(tmp_path: Path) -> None:
    project(tmp_path)
    (tmp_path / "ledger/errors.py").write_text(
        "from .service import f\n\n\nclass ValidationError(Exception):\n    pass\n")
    assert "is defined in" not in replace(tmp_path)


def test_the_projects_package_used_as_a_module_gets_its_import(tmp_path: Path) -> None:
    # tasks 5 on 10-28: `raise tasktracker.errors.NotFoundError(...)` refused 48 times
    result = replace(project(tmp_path), USE.replace("ValidationError", "ledger.errors.ValidationError"))
    assert "`ledger` is defined in ledger/errors.py: add `import ledger.errors`" in result


def test_switched_off_the_refusal_is_as_before(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "import_offer")
    result = replace(project(tmp_path))
    assert "is defined in" not in result
    assert "define or import each undefined name in the same write" in result


def test_apply_is_off_by_default() -> None:
    assert "import_offer_apply" in features.OPT_IN
    assert not features.on("import_offer_apply")


def test_apply_adds_the_import_and_saves_the_write(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "import_offer_apply")
    result = replace(project(tmp_path))
    assert result.startswith("Replaced definition f in ledger/service.py")
    assert "(Vega added from .errors import ValidationError: defined in ledger/errors.py.)" \
        in result
    saved = (tmp_path / "ledger/service.py").read_text()
    assert "from .errors import ValidationError\n" in saved
    assert "raise ValidationError('no')" in saved


def test_apply_refuses_when_another_name_has_no_home(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "import_offer_apply")
    result = replace(project(tmp_path), USE.replace("return x", "return Missing()"))
    assert result.startswith("Write not applied")
    assert "`ValidationError` is defined in ledger/errors.py" in result
    assert "ValidationError" not in (tmp_path / "ledger/service.py").read_text()
