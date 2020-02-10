from .entropy import entro_ave, footer_entro, entro_post_analysis
from .entropy import footer_entro_analyse


_routines_dict = {
    #'get_entro_ave_post': [_entro_ave_postprocessed, 'Entropy', 13,
    #                      footer_entro_post],
    'get_entro_ave': [entro_ave, 'Entropy', 17, footer_entro],
    'entro_analyse': [entro_post_analysis, 'Entropy',
                      None, footer_entro_analyse]
    #'get_r': [_get_r, 'r_data', 6, footer_r]

}
