#
#  tfutils - A set of tools for training networks with tensorflow
#  Copyright (C) 2017  Benjamin Ummenhofer
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Module with some easing functions"""
import tensorflow as tf
import numpy as np


def interpolate_linear( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(change_value*t + start_value)


def ease_in_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.cast(change_value*t*t + start_value,tf.float32)


def ease_out_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_out_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.cast(-change_value*t*(t-2) + start_value,tf.float32)


def ease_in_out_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_out_quad"):
        d_2 = 0.5*duration
        c_2 = 0.5*change_value
        return tf.cond( current_time/duration < 0.5, 
                lambda:ease_in_quad(current_time, start_value, c_2, d_2), 
                lambda:ease_out_quad(current_time-d_2, start_value+c_2, c_2, d_2) )


