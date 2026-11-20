"""One frozen test recipe, immutable identity and no-overwrite CLI guards."""
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_score_test import run,state
from wbm_retrieval.common import read_json,write_json


class RunGuards(unittest.TestCase):
    def test_dry_run_is_one_fixed_test_recipe_without_io(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)),\
                patch.object(run.inputs,'load',side_effect=AssertionError('source IO')),redirect_stdout(StringIO()) as captured:
            self.assertEqual(run.main(['--dry-run']),0)
            value=json.loads(captured.getvalue())
            self.assertEqual(value['split'],'test')
            self.assertEqual(value['variant'],'mixed_a0p5_e1p0')
            self.assertEqual(value['spec']['alpha'],.5)
            self.assertEqual(value['spec']['eta'],1.)
            self.assertFalse(value['test_parameter_search'])
            self.assertFalse(value['training'])
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_cli_does_not_accept_other_stages_or_hyperparameters(self):
        with redirect_stderr(StringIO()):
            for args in (['--stage','valid'],['--alpha','.75'],['--eta','.5'],['--variant','product']):
                with self.assertRaises(SystemExit):run.main(args)

    def test_old_nested_external_outputs_and_bad_workers_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)):
            p=Path(tmp)/'artificial/old';p.mkdir(parents=True)
            write_json(p/'run_manifest.json',{'fixture':True})
            for name in ('artificial/old','artificial/old/nested','artificial','artifacts/new'):
                with self.assertRaises(ValueError):run.main(['--dry-run','--output-root',name])
            for n in ('0','9'):
                with self.assertRaises(ValueError):run.main(['--dry-run','--workers',n])
            with self.assertRaises(ValueError):run.main(['--dry-run','--resume-root','artificial/missing'])

    def test_identity_is_test_and_cannot_be_reassigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);first=state.identity(p,'mixed_a0p5_e1p0','sig')
            self.assertEqual(first['split'],'test')
            self.assertTrue(first['key'].startswith('f_region_score_test_'))
            self.assertEqual(first,state.identity(p,'mixed_a0p5_e1p0','sig'))
            with self.assertRaises(ValueError):state.identity(p,'mixed_a0p5_e1p0','other')
            path=p/'identities/mixed_a0p5_e1p0.json';value=read_json(path)
            value['started_at']=value['started_at'].replace('+08:00','+00:00');write_json(path,value)
            with self.assertRaises(ValueError):state.identity(p,'mixed_a0p5_e1p0','sig')

    def test_resume_cannot_write_into_a_sealed_previous_experiment(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)):
            p=Path(tmp)/'artificial/old';p.mkdir(parents=True)
            write_json(p/'run_manifest.json',state.signed({'version':'f_region_score_search_v1','requested_stage':'valid'}))
            before={str(f.relative_to(p)):f.read_bytes() for f in p.rglob('*') if f.is_file()}
            with patch.object(run.inputs,'load',side_effect=AssertionError('must reject before input loading')):
                with self.assertRaises(ValueError):run.main(['--resume-root','artificial/old'])
            self.assertEqual(before,{str(f.relative_to(p)):f.read_bytes() for f in p.rglob('*') if f.is_file()})


if __name__=='__main__':unittest.main()
