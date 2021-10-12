from pydantic import main
import scgenerator as sc


def test_descriptor():
    # Same branch
    var1 = sc.VariationDescriptor(
        raw_descr=[[("num", 1), ("a", False)], [("b", 0)]], index=[[1, 0], [0]]
    )
    var2 = sc.VariationDescriptor(
        raw_descr=[[("num", 2), ("a", False)], [("b", 0)]], index=[[1, 0], [0]]
    )
    assert var1.branch.identifier == "b_0"
    assert var1.identifier != var1.branch.identifier
    assert var1.identifier != var2.identifier
    assert var1.branch.identifier == var2.branch.identifier

    # different branch
    var3 = sc.VariationDescriptor(
        raw_descr=[[("num", 2), ("a", True)], [("b", 0)]], index=[[1, 0], [0]]
    )
    assert var1.branch.identifier != var3.branch.identifier
    assert var1.formatted_descriptor() != var2.formatted_descriptor()
    assert var1.formatted_descriptor() != var3.formatted_descriptor()


def test_variationer():
    var = sc.Variationer(
        [
            dict(a=[1, 2], num=[0, 1, 2]),
            [dict(b=["000", "111"], c=["a", "-1"])],
            dict(),
            dict(),
            [dict(aaa=[True, False], bb=[1, 3])],
        ]
    )
    assert var.var_num(0) == 6
    assert var.var_num(1) == 12
    assert var.var_num() == 24

    cfg = dict(bb=None)
    branches = set()
    for descr in var.iterate():
        assert descr.update_config(cfg).items() >= set(descr.raw_descr[-1])
        branches.add(descr.branch.identifier)
    assert len(branches) == 8


def main():
    test_descriptor()


if __name__ == "__main__":
    main()
