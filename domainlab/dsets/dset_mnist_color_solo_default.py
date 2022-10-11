from domainlab.dsets.a_dset_mnist_color_rgb_solo import ADsetMNISTColorRGBSolo
from domainlab.dsets.utils_color_palette import default_rgb_palette


class DsetMNISTColorSoloDefault(ADsetMNISTColorRGBSolo):
    @property
    def palette(self):
        return default_rgb_palette

    def get_num_colors(self):
        return len(self.palette)

    def get_background_color(self, ind):
        if self.color_scheme == 'back':
            return self.palette[ind]
        if self.color_scheme == 'both':
            return self.palette[-(ind-3)]
        # only array can be multiplied with number 255 directly
        return self.palette[ind]
        # "num" do not use background at all

    def get_foreground_color(self, ind):
        if self.color_scheme == "num":
            return self.palette[-(ind + 1)]
        return self.palette[ind]
