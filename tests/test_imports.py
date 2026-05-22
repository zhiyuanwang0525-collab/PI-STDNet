def test_import_package():
    import pytest

    pytest.importorskip("torch")
    import pistdnet
    from pistdnet.models import PI_STDNet

    assert pistdnet.__version__
    assert PI_STDNet is not None
