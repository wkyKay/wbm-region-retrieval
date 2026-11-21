"""Independent full-pool identities, frozen valid counts and undefined metrics."""
import csv
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from f_region_wbbs import protocol
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def fixture(split="valid"):
    ids=np.arange(2,16,2,dtype=np.int64)
    labels=np.zeros((16,8),dtype=np.uint8)
    labels[ids]=[[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0],[1,1,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0]]
    pool=FullGalleryPool(ids[[0,3,5,6]],ids,split,"synthetic")
    return pool,labels


class ProtocolTests(unittest.TestCase):
    def test_cross_valid_keeps_queries_and_gallery_and_excludes_complete_combinations(self):
        base,labels=fixture();before=labels.copy()
        cross=protocol.make_cross_pool(base,labels)
        receipt=protocol.audit_pool(cross,labels,ks=[1,3],formal=False)
        np.testing.assert_array_equal(labels,before)
        np.testing.assert_array_equal(list(cross),list(base))
        np.testing.assert_array_equal(cross.gallery,base.gallery)
        self.assertEqual(cross[2].tolist(),[6,8,10,12,14])
        self.assertIn(12,cross[8])  # proper supersets remain eligible
        self.assertEqual(receipt["full_candidates_checked"],23)
        self.assertEqual(receipt["metric_query_counts"]["Coverage-nDCG@1"],3)
        self.assertEqual(receipt["metric_query_counts"]["FullCoverage_mAP@1"],2)
        self.assertEqual(receipt["metric_query_counts"]["FullCoveragePrecision@1"],4)
        self.assertEqual(receipt["groups"]["by_label_count"]["3"]["queries_without_full_coverage_candidates"],1)

    def test_ordinary_audit_includes_same_combination_and_excludes_only_self(self):
        base,labels=fixture()
        receipt=protocol.audit_pool(base,labels,ks=[1,3],formal=False)
        self.assertEqual(receipt["full_candidates_checked"],24)
        self.assertEqual(receipt["protocol"],"retrieval_v1")
        self.assertIn(4,base[2])

    def test_forged_metadata_candidate_subsets_and_order_are_rejected(self):
        for issue in ("hash","size","signature","pool","order"):
            with self.subTest(issue=issue):
                base,labels=fixture();cross=protocol.make_cross_pool(base,labels)
                if issue=="hash":cross.metadata["candidate_identity_sha256"]="a"*64
                elif issue=="size":cross.metadata["candidate_count_min"]+=1
                elif issue=="signature":cross.protocol_signature="different"
                else:
                    original=cross
                    class Broken:
                        split_name=base.split_name;gallery=base.gallery
                        metadata=cross.metadata;protocol_signature=cross.protocol_signature
                        candidates_per_query=cross.candidates_per_query
                        def __iter__(self):return iter(original)
                        def __getitem__(self,q):return original[q][::-1] if issue=="order" else original[q][1:]
                    cross=Broken()
                with self.assertRaises(ValueError):protocol.audit_pool(cross,labels,[1,3],formal=False)

    def test_formal_flag_cannot_promote_small_fixture(self):
        base,labels=fixture()
        with self.assertRaisesRegex(ValueError,"Formal"):protocol.audit_pool(protocol.make_cross_pool(base,labels),labels)

    def test_actual_frozen_cross_valid_counts_and_hashes_are_independent(self):
        root=Path(__file__).resolve().parents[2]
        with (root/"protocol/deduplicated/split.csv").open() as f:rows=list(csv.DictReader(f))
        labels=np.zeros((max(int(r["sample_id"]) for r in rows)+1,8),dtype=np.uint8)
        for row in rows:labels[int(row["sample_id"])]=[int(row[f"label_{i}"]) for i in range(8)]
        def ids(name):
            with (root/"protocol/retrieval_v1"/name).open() as f:return np.asarray([int(r["sample_id"]) for r in csv.DictReader(f)])
        base=FullGalleryPool(ids("valid_queries.csv"),ids("valid_gallery.csv"),"valid",protocol.BASE_PROTOCOL_SIGNATURE)
        cross=protocol.make_cross_pool(base,labels)
        result=protocol.audit_pool(cross,labels)
        self.assertEqual(result["candidate_identity_sha256"],protocol.FROZEN["valid"]["candidate_identity_sha256"])
        self.assertEqual(result["full_candidates_checked"],1750095)
        self.assertEqual((result["candidate_count_min"],result["candidate_count_max"]),(3499,3585))
        self.assertEqual(result["metric_query_counts"]["Coverage-nDCG@10"],484)
        self.assertEqual(result["metric_query_counts"]["FullCoverage_mAP@10"],428)
        self.assertEqual({k:v["num_queries"] for k,v in result["groups"]["by_label_count"].items()},
                         {"1":98,"2":179,"3":167,"4":56})


if __name__=="__main__":unittest.main()
