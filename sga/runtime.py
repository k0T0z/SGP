#################################################################################
#                                                                               #
#  Copyright (C) 2026 Seif Kandil (k0T0z)                                       #
#                                                                               #
#  This file is a part of the ENIGMA Development Environment.                   #
#                                                                               #
#                                                                               #
#  ENIGMA is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software    #
#  Foundation, version 3 of the license or any later version.                   #
#                                                                               #
#  This application and its source code is distributed AS-IS, WITHOUT ANY       #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS    #
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more        #
#  details.                                                                     #
#                                                                               #
#  You should have recieved a copy of the GNU General Public License along      #
#  with this code. If not, see <http://www.gnu.org/licenses/>                   #
#                                                                               #
#  ENIGMA is an environment designed to create games and other programs with a  #
#  high-level, fully compilable language. Developers of ENIGMA or anything      #
#  associated with ENIGMA are in no way responsible for its users or            #
#  applications created by its users, or damages caused by the environment      #
#  or programs made in the environment.                                         #
#                                                                               #
#################################################################################

"""Runtime access to the ``shader_gen_py`` extension module.

The extension is built into ``build/python`` next to the protobuf/abseil DLLs it links
against, so that directory has to be registered with the loader before the import. Doing it
here means every module that needs the extension gets it from one place.
"""

import os
import sys
from pathlib import Path

_BUILD_PYTHON = Path(__file__).resolve().parent.parent / "build" / "python"
if _BUILD_PYTHON.is_dir():
    os.add_dll_directory(str(_BUILD_PYTHON))
    sys.path.insert(0, str(_BUILD_PYTHON))

import shader_gen_py  # noqa: E402
