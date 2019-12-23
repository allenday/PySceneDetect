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

""" Module: ``scenedetect.detectors.blur_detector``

This module implements the :py:class:`BlurDetector`, which uses a set intensity
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

# PySceneDetect Library Imports
from scenedetect.scene_detector import SceneDetector


##
## BlurDetector Helper Functions
##

def compute_frame_chisq(channel,frame1,frame0):
    mat0 = None
    mat1 = None
    if frame1 is None:
        return -99
    if channel == 'r':
        mat0 = frame0[:,:,0]
        mat1 = frame1[:,:,0]
    elif channel == 'g':
        mat0 = frame0[:,:,1]
        mat1 = frame1[:,:,1]
    elif channel == 'b':
        mat0 = frame0[:,:,2]
        mat1 = frame1[:,:,2]
    elif channel == 's':
        mat0 = frame0[:,:,:]
        mat1 = frame1[:,:,:]
    hist0 = _hist(mat0)
    hist1 = _hist(mat1)
    results = scipy.stats.chisquare(hist0,hist1)
    p = results[1]
    if math.isnan(p):
        return -99
    elif p == 0:
        return 100
    return -1*math.log10(p)
    

def compute_frame_hist_s(frame):
    hist = ( _hist(frame[:,:,0]) + _hist(frame[:,:,1]) + _hist(frame[:,:,2]) ) / 3
    return hist
def _hist(channel):
    hist = numpy.floor(100*numpy.histogram(channel.flatten(),bins=[0,31,63,95,127,159,191,223,255])[0]/channel.flatten().size)
    return hist

def compute_frame_z(frame):
    if frame is None:
        return 999
    z = numpy.std(frame[:, :, :]) / numpy.mean(frame[:, :, :])
    return z

def compute_frame_min(frame):
    return numpy.min(frame[:,:,0]+frame[:,:,1]+frame[:,:,2])

def compute_frame_max(frame):
    return numpy.max(frame[:,:,0]+frame[:,:,1]+frame[:,:,2])

def compute_frame_p25(frame):
    return numpy.percentile(frame[:,:,0]+frame[:,:,1]+frame[:,:,2],25)

def compute_frame_p75(frame):
    return numpy.percentile(frame[:,:,0]+frame[:,:,1]+frame[:,:,2],75)

def compute_frame_range(frame):
    return numpy.max(frame[0] + frame[1] + frame[2]) - numpy.min(frame[0] + frame[1] + frame[2])

def compute_frame_average(frame):
    """Computes the average pixel value/intensity for all pixels in a frame.

    The value is computed by adding up the 8-bit R, G, and B values for
    each pixel, and dividing by the number of pixels multiplied by 3.

    Returns:
        Floating point value representing average pixel intensity.
    """
    num_pixel_values = float(
        frame.shape[0] * frame.shape[1] * frame.shape[2])
    avg_pixel_value = numpy.sum(frame[:, :, :]) / num_pixel_values
    return avg_pixel_value


##
## BlurDetector Class Implementation
##

class BlurDetector(SceneDetector):
    """Detects fast cuts/slow fades in from and out to a given threshold level.

    Detects both fast cuts and slow fades so long as an appropriate threshold
    is chosen (especially taking into account the minimum grey/black level).

    Attributes:
        threshold:  Z-score of 8-bit intensity value of all pixel values (R, G, and B)
            must be <= to in order to trigger a blur transition.
        min_scene_len:  FrameTimecode object or integer greater than 0 of the
            minimum length, in frames, of a scene (or subsequent scene cut).
        fade_bias:  Float between -1.0 and +1.0 representing the percentage of
            timecode skew for the start of a scene (-1.0 causing a cut at the
            fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be
            right at the position where the threshold is passed).
        add_final_scene:  Boolean indicating if the video ends on a fade-out to
            generate an additional scene at this timecode.
    """
    def __init__(self, threshold=12, min_scene_len=15,
                 fade_bias=0.0, add_final_scene=False):
        """Initializes threshold-based scene detector object."""

        super(BlurDetector, self).__init__()
        self.threshold = threshold
        self.fade_bias = fade_bias
        self.min_scene_len = min_scene_len
        self.last_frame_avg   = None
        self.last_scene_cut   = None
        self.last_frame_img   = None
        # Whether to add an additional scene or not when ending on a fade out
        # (as cuts are only added on fade ins; see post_process() for details).
        self.add_final_scene = add_final_scene
        # Where the last fade (threshold crossing) was detected.
        self.last_fade = {
            'frame': 0,         # frame number where the last detected fade is
            'type': None        # type of fade, can be either 'in' or 'out'
        }
        self._metric_keys = ['delta_rgb','z_rgb','p25_rgb','p75_rgb','min_rgb','max_rgb','hist_r','hist_g','hist_b','hist_s']
        self.cli_name = 'detect-blur'

    def frame_under_threshold(self, frame):
        """Check if the frame is below (true) or above (false) the threshold.

        Instead of using the average, we check all pixel values (R, G, and B)
        meet the given threshold (within the minimum percent).  This ensures
        that the threshold is not exceeded while maintaining some tolerance for
        compression and noise.

        This is the algorithm used for absolute mode of the threshold detector.

        Returns:
            Boolean, True if the number of pixels whose R, G, and B values are
            all <= the threshold, or False if not.
        """
        z = compute_frame_z(frame)
        if z < self.threshold:
            return True
        return False

    def process_frame(self, frame_num, frame_img):
        # type: (int, Optional[numpy.ndarray]) -> List[int]
        """
        Args:
            frame_num (int): Frame number of frame that is being passed.
            frame_img (numpy.ndarray or None): Decoded frame image (numpy.ndarray) to perform
                scene detection with. Can be None *only* if the self.is_processing_required()
                method (inhereted from the base SceneDetector class) returns True.
        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self.last_scene_cut is None:
            self.last_scene_cut = frame_num

        # Compare the # of pixels under threshold in current_frame & last_frame.
        # If absolute value of pixel intensity delta is above the threshold,
        # then we trigger a new scene cut/break.

        # List of cuts to return.
        cut_list = []

        # The metric used here to detect scene breaks is the percent of pixels
        # less than or equal to the threshold; however, since this differs on
        # user-supplied values, we supply the average pixel intensity as this
        # frame metric instead (to assist with manually selecting a threshold)
        frame_avg = 0.0
        frame_z = 0.0
        frame_range = 0.0
        frame_hist_r = []
        frame_hist_g = []
        frame_hist_b = []
        frame_hist_s = []

        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num, self._metric_keys)):
            frame_avg    = self.stats_manager.get_metrics(frame_num, self._metric_keys)[0]
            frame_z      = self.stats_manager.get_metrics(frame_num, self._metric_keys)[1]
            frame_p25    = self.stats_manager.get_metrics(frame_num, self._metric_keys)[2]
            frame_p75    = self.stats_manager.get_metrics(frame_num, self._metric_keys)[3]
            frame_min    = self.stats_manager.get_metrics(frame_num, self._metric_keys)[4]
            frame_max    = self.stats_manager.get_metrics(frame_num, self._metric_keys)[5]
            frame_hist_r = self.stats_manager.get_metrics(frame_num, self._metric_keys)[6]
            frame_hist_g = self.stats_manager.get_metrics(frame_num, self._metric_keys)[7]
            frame_hist_b = self.stats_manager.get_metrics(frame_num, self._metric_keys)[8]
            frame_hist_s = self.stats_manager.get_metrics(frame_num, self._metric_keys)[9]
        else:
            frame_avg    = compute_frame_average(frame_img)
            frame_z      = compute_frame_z(frame_img)
            frame_p25    = compute_frame_p25(frame_img)
            frame_p75    = compute_frame_p75(frame_img)
            frame_min    = compute_frame_min(frame_img)
            frame_max    = compute_frame_max(frame_img)
            frame_chisq_r = compute_frame_chisq('r', self.last_frame_img, frame_img)
            frame_chisq_g = compute_frame_chisq('g', self.last_frame_img, frame_img)
            frame_chisq_b = compute_frame_chisq('b', self.last_frame_img, frame_img)
            frame_chisq_s = compute_frame_chisq('s', self.last_frame_img, frame_img)
            if self.stats_manager is not None:
                self.stats_manager.set_metrics(frame_num, {
                    self._metric_keys[0]: frame_avg,
                    self._metric_keys[1]: frame_z,
                    self._metric_keys[2]: frame_p25,
                    self._metric_keys[3]: frame_p75,
                    self._metric_keys[4]: frame_min,
                    self._metric_keys[5]: frame_max,
                    self._metric_keys[6]: frame_chisq_r,
                    self._metric_keys[7]: frame_chisq_g,
                    self._metric_keys[8]: frame_chisq_b,
                    self._metric_keys[9]: frame_chisq_s
                })

        if self.last_frame_avg is not None:
            if self.last_fade['type'] == 'in' and self.frame_under_threshold(frame_img):
                # Just faded out of a scene, wait for next fade in.
                self.last_fade['type'] = 'out'
                self.last_fade['frame'] = frame_num
            elif self.last_fade['type'] == 'out' and not self.frame_under_threshold(frame_img):
                # Only add the scene if min_scene_len frames have passed.
                if (frame_num - self.last_scene_cut) >= self.min_scene_len:
                    # Just faded into a new scene, compute timecode for the scene
                    # split based on the fade bias.
                    f_out = self.last_fade['frame']
                    f_split = int((frame_num + f_out +
                                   int(self.fade_bias * (frame_num - f_out))) / 2)
                    #cut_list.append(f_split)
                    self.last_scene_cut = frame_num
                self.last_fade['type'] = 'in'
                self.last_fade['frame'] = frame_num
        else:
            self.last_fade['frame'] = 0
            if self.frame_under_threshold(frame_img):
                self.last_fade['type'] = 'out'
            else:
                self.last_fade['type'] = 'in'
        # Before returning, we keep track of the last frame average (can also
        # be used to compute fades independently of the last fade type).
        self.last_frame_avg = frame_avg
        self.last_frame_img = frame_img
        return cut_list

    def post_process(self, frame_num, frame_rate):
        """Writes a final scene cut if the last detected fade was a fade-out.

        Only writes the scene cut if add_final_scene is true, and the last fade
        that was detected was a fade-out.  There is no bias applied to this cut
        (since there is no corresponding fade-in) so it will be located at the
        exact frame where the fade-out crossed the detection threshold.
        """
        cut_times = []
        if self.last_fade['type'] == 'out' and self.add_final_scene and (
                (self.last_scene_cut is None and frame_num >= self.min_scene_len) or
                (frame_num - self.last_scene_cut) >= self.min_scene_len):
            cut_times.append(self.last_fade['frame'])
        return cut_times
