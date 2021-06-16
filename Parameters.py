# class for parameters passed through methods


class Parameters:

    def __init__(self, markov_threshold=0.75, fc_threshold=1.0, fc_context_num=3, fc_segmentation_level=3):
        # markov
        self.mkv_thr = markov_threshold
        # form classes
        self.fc_thr = fc_threshold
        self.fc_n_ctx = fc_context_num
        self.fc_seg_ord = fc_segmentation_level
