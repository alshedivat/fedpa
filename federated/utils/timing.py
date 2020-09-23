# coding=utf-8
# Copyright 2020 Maruan Al-Shedivat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for timing."""

import timeit


class Timer(object):
    def __init__(self, description):
        self.description = description
        self.elapsed = None

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, type, value, traceback):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
