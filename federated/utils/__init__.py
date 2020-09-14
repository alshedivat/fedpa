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

import functools


def enforce_kwargs(positional_args):
    """Enforces passing arguments to the function by kwargs only.

    Args:
        positional_args: A tuple of argument names that are passed to the
            function in the specified order as positional arguments, followed
            by any additional keyword arguments.
    """

    def _enforce_kwargs(func):
        @functools.wraps(func)
        def _wrapped_func(**kwargs):
            args = []
            for name in positional_args:
                args.append(kwargs.pop(name))
            return func(*args, **kwargs)

        return _wrapped_func

    return _enforce_kwargs
