from state._cli._tx._train import decoder_required_for_output_space


def test_decoder_not_required_for_direct_hvg_gene_training():
    assert decoder_required_for_output_space(None, "gene") is False
    assert decoder_required_for_output_space("X_hvg", "gene") is False


def test_decoder_required_for_embedding_to_gene_training():
    assert decoder_required_for_output_space("X_demo", "gene") is True


def test_decoder_required_for_embedding_to_all_gene_training():
    assert decoder_required_for_output_space("X_hvg", "all") is True
    assert decoder_required_for_output_space("X_demo", "all") is True


def test_decoder_not_required_without_embedding_input_for_all_space():
    assert decoder_required_for_output_space(None, "all") is False
