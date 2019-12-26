# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2019 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

""" Module: ``scenedetect.detectors.rupture_detector``

This module implements the :py:class:`RuptureDetector`, which uses a set intensity
level as a threshold, to detect cuts when the average frame intensity exceeds or falls
below this threshold.

This detector is available from the command-line interface by using the
`detect-threshold` command.
"""

# Third-Party Library Imports
import logging
import math
import numpy
import pandas
import ruptures as rpt
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer



# PySceneDetect Library Imports
from scenedetect.scene_detector import SceneDetector

##
## RuptureDetector Class Implementation
##

class RuptureDetector(SceneDetector):
    """Detects fast cuts/slow fades in from and out to a given threshold level.

    Detects both fast cuts and slow fades so long as an appropriate threshold
    is chosen (especially taking into account the minimum grey/black level).

    Attributes:
        threshold:  Z-score of 8-bit intensity value of all pixel values (R, G, and B)
            must be <= to in order to trigger a rupture changepoint transition.
        min_scene_len:  FrameTimecode object or integer greater than 0 of the
            minimum length, in frames, of a scene (or subsequent scene cut).
        fade_bias:  Float between -1.0 and +1.0 representing the percentage of
            timecode skew for the start of a scene (-1.0 causing a cut at the
            fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be
            right at the position where the threshold is passed).
        add_final_scene:  Boolean indicating if the video ends on a fade-out to
            generate an additional scene at this timecode.
    """
    def __init__(self, threshold=12, min_scene_len=15):
        """Initializes threshold-based scene detector object."""

        super(RuptureDetector, self).__init__()
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self._metric_keys = [
        #        'content_val', 'delta_hue', 'delta_sat', 'delta_lum','delta_rgb', 'z_rgb', 'p25_rgb', 'p75_rgb', 'min_rgb', 'max_rgb', 'hist_r', 'hist_g', 'hist_b', 'hist_s'
            ]
        self.cli_name = 'detect-rupture'

    def frame_under_threshold(self, frame):
        return False

    def process_frame(self, frame_num, frame_img):
        return []

    def post_process(self, frame_num, frame_rate):

        imputer = Imputer()

        logging.info('****RuptureDetector post-process')
        logging.info("**** stats metrics: " + str(self.stats_manager.get_registered_metrics()))
        cut_times = []

        content_val = self.stats_manager.get_metric('content_val')
        xr = numpy.array(range(0,len(content_val))).reshape(-1,1)
        content_vali = imputer.fit_transform(numpy.array(content_val).reshape(-1,1))

        content_val_lm = LinearRegression().fit(xr,numpy.cumsum(content_vali))
        content_val2 = numpy.cumsum(content_vali) - content_val_lm.predict(xr)

        delta_lum   = self.stats_manager.get_metric('delta_lum')
        delta_lumi = imputer.fit_transform(numpy.array(delta_lum).reshape(-1,1))
        delta_lum_lm = LinearRegression().fit(xr,numpy.cumsum(delta_lumi))
        delta_lum2 = numpy.cumsum(delta_lumi) - delta_lum_lm.predict(xr)

        delta_hue   = self.stats_manager.get_metric('delta_hue')
        delta_huei = imputer.fit_transform(numpy.array(delta_hue).reshape(-1,1))
        delta_hue_lm = LinearRegression().fit(xr,numpy.cumsum(delta_huei))
        delta_hue2 = numpy.cumsum(delta_huei) - delta_hue_lm.predict(xr)

        delta_sat   = self.stats_manager.get_metric('delta_sat')
        delta_sati = imputer.fit_transform(numpy.array(delta_sat).reshape(-1,1))
        delta_sat_lm = LinearRegression().fit(xr,numpy.cumsum(delta_sati))
        delta_sat2 = numpy.cumsum(delta_sati) - delta_sat_lm.predict(xr)

        delta_rgb   = self.stats_manager.get_metric('delta_rgb')
        delta_rgbi = imputer.fit_transform(numpy.array(delta_rgb).reshape(-1,1))
        delta_rgb_lm = LinearRegression().fit(xr,numpy.cumsum(delta_rgbi))
        delta_rgb2 = numpy.cumsum(delta_rgbi) - delta_rgb_lm.predict(xr)

        z_rgb       = self.stats_manager.get_metric('z_rgb')
        z_rgbi = imputer.fit_transform(numpy.array(z_rgb).reshape(-1,1))
        z_rgb_lm = LinearRegression().fit(xr,numpy.cumsum(z_rgbi))
        z_rgb2 = numpy.cumsum(z_rgbi) - z_rgb_lm.predict(xr)

        p25_rgb     = self.stats_manager.get_metric('p25_rgb')
        p75_rgb     = self.stats_manager.get_metric('p75_rgb')
        iqr_rgb = []
        for i in range(0,len(p25_rgb)):
            iqr_rgb.append(p75_rgb[i] - p25_rgb[i])
        iqr_rgbi = imputer.fit_transform(numpy.array(iqr_rgb).reshape(-1,1))
        iqr_rgb_lm = LinearRegression().fit(xr,numpy.cumsum(iqr_rgbi))
        iqr_rgb2 = numpy.cumsum(iqr_rgbi) - iqr_rgb_lm.predict(xr)

        #chi_s       = self.stats_manager.get_metric('hist_s')
        #ts          = pandas.Series(self.stats_manager.get_metric('z_rgb'))

        signal = numpy.column_stack((content_val2, delta_lum2, delta_hue2, delta_sat2, delta_rgb, delta_rgb2, z_rgb2, iqr_rgb2))
       
        jmp = int(frame_rate/6)
        #margin = int(frame_rate*20)
        margin = 0
        fps = int(math.ceil(frame_rate))
        algo = rpt.Pelt(model="rbf",min_size=int(self.min_scene_len),jump=jmp).fit(signal[margin:(signal.shape[0]-margin),])
        logging.info("****"+str(self.threshold))
        result = algo.predict(pen=self.threshold)
 
        cut_times = result
        return cut_times
