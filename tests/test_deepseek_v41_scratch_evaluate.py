import json
import tempfile
from pathlib import Path
import unittest
from archlab.automodel.deepseek_v41_scratch_evaluate import checkpoint_marker,choice_result,json_lines


class ScratchEvalTests(unittest.TestCase):
    def test_checkpoint_requires_matched_variant_tokens_and_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m={'format':'archlab-v41-full-sharded-v1','world_size':8,'cursor':{'supervised_tokens':240875236},'contract':{'format':'archlab-v41-scratch-comparison-v1','world_size':8,'variant':'normal'}}
            (p/'COMPLETE.json').write_text(json.dumps(m))
            self.assertEqual(checkpoint_marker(p,'normal',240875236),m)
            for v,t in [('simplicial',240875236),('normal',240875237)]:
                with self.assertRaises(ValueError):checkpoint_marker(p,v,t)
            m['world_size']=16;(p/'COMPLETE.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):checkpoint_marker(p,'normal',240875236)

    def test_unicode_line_separator_inside_json_string(self):
        records=[{'question':'first\u2028second\u0085third'}, {'question':'next'}]
        text='\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n'
        self.assertEqual(json_lines(text),records)

    def test_choice_character_normalization(self):
        c={'task':'piqa','choices':['aa','aaaaaaaaaa'],'answer':1}
        r=choice_result(c,[-2.,-3.]);self.assertEqual(r['prediction'],0);self.assertEqual(r['normalized_prediction'],1);self.assertEqual(r['accuracy_norm'],1)

    def test_mmlu_scores_answer_letters(self):
        c={'task':'mmlu','choices':['aa','aaaaaaaaaa'],'answer':0}
        r=choice_result(c,[-2.,-3.]);self.assertEqual(r['prediction'],r['normalized_prediction']);self.assertEqual(r['accuracy_norm'],1)

    def test_incomplete_or_nonfinite_scores_rejected(self):
        c={'task':'arc_challenge','choices':['a','b'],'answer':0}
        for scores in [[None,0.], [float('nan'),0.], [0.]]:
            with self.assertRaises(ValueError):choice_result(c,scores)


if __name__=='__main__':unittest.main()
