from libdg.compos.pcr.request import RequestVAEBuilderCHW
from libdg.compos.vae.utils_request_chain_builder import VAEChainNodeGetter


def test_VAEChainBuilderScenarioAsRequest():
    request = RequestVAEBuilderCHW(3, 64, 64, None)
    node = VAEChainNodeGetter(request)()
    builder = node.init_business(8, 8, 8)
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
