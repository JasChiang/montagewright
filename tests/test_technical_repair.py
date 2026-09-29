import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright import technical_repair as repair


def wiring_error():
    try:
        exec(compile('missing_wire()', '/project/src/montagewright/cli.py', 'exec'))
    except NameError as error:
        return error


def test_budget_and_semantic_failures_never_start_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(repair.shutil, 'which', lambda _: pytest.fail('must not launch'))
    assert repair.attempt(ValueError('wrong subject'), output=tmp_path, argv=[]) is None
    monkeypatch.setenv(repair.ATTEMPT_ENV, '1')
    assert repair.attempt(wiring_error(), output=tmp_path, argv=[]) is None


@pytest.mark.parametrize('validation_code,changed_file', [(0, 'cli.py'), (1, 'cli.py'), (0, 'cost.py')])
def test_only_validated_isolated_wiring_patch_resumes(tmp_path, monkeypatch, validation_code, changed_file):
    repository = tmp_path / 'repo'
    src = repository / 'src' / 'montagewright'
    src.mkdir(parents=True)
    for name in ('cli.py', 'cost.py'):
        (src / name).write_text('value = 1\n')
    (repository / 'tests').mkdir()
    (repository / 'tests' / 'test_original.py').write_text('assert True\n')
    (repository / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
    output = tmp_path / 'output'
    calls = []
    monkeypatch.delenv(repair.ATTEMPT_ENV, raising=False)
    monkeypatch.setattr(repair, 'codex_executable', lambda: '/bin/codex')
    def run(command, **kwargs):
        calls.append(command)
        if 'exec' in command:
            stage = Path(command[command.index('-C') + 1])
            assert stage != repository
            (stage / 'src' / 'montagewright' / changed_file).write_text('value = 2\n')
            (stage / 'tests' / 'test_original.py').write_text('assert False\n')
            (stage / 'answer.json').write_text(json.dumps({'fixed': True, 'explanation': 'correct missing wire'}))
            return SimpleNamespace(returncode=0)
        if 'pytest' in command:
            assert (Path(kwargs['cwd']) / 'tests' / 'test_original.py').read_text() == 'assert True\n'
            assert (src / 'cli.py').read_text() == 'value = 1\n'
            return SimpleNamespace(returncode=validation_code)
        assert kwargs['env'][repair.ATTEMPT_ENV] == '1'
        return SimpleNamespace(returncode=75)
    monkeypatch.setattr(repair.subprocess, 'run', run)
    result = repair.attempt(wiring_error(), output=output, argv=['render', 'rushes'], repository=repository)
    succeeds = validation_code == 0 and changed_file == 'cli.py'
    assert result == (75 if succeeds else None)
    assert (src / 'cli.py').read_text() == ('value = 2\n' if succeeds else 'value = 1\n')
    assert (src / 'cost.py').read_text() == 'value = 1\n'
    assert len(calls) == (3 if succeeds else 2 if changed_file == 'cli.py' else 1)
    before = len(calls)
    assert repair.attempt(wiring_error(), output=output, argv=[], repository=repository) is None
    assert len(calls) == before
