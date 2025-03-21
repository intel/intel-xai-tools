#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

COMMAND=${1}

# Temp directory within docs
TEMP_DIR="markdown"

if [[ ${COMMAND} == "clean" ]]; then
    rm -rf ${TEMP_DIR}
elif [[ ${COMMAND} == "html" ]]; then
    # Create a temp directory for markdown files that are just used for sphinx docs
    mkdir -p ${TEMP_DIR}

    # This script takes sections out of the main README.md to create smaller .md files that are used for pages
    # in the sphinx doc table of contents (like Overview, Installation, Legal Information).
    # If heading name changes are made in the main README.md, they will need to be updated here too because the sed
    # commands are grabbing the text between two headers.

    # We don't want to mess with the original README.md, so create a copy of it before we start editing
    cp ../README.md ${TEMP_DIR}/Welcome.md

    # Convert links to go to sphinx docs
    sed -i 's#DATASETS.md#<datasets>#g' ${TEMP_DIR}/Welcome.md
    sed -i 's#MODELS.md#<models>#g' ${TEMP_DIR}/Welcome.md
    sed -i 's#notebooks\#model-card-generator-tutorial-notebooks#<notebooks>#g' ${TEMP_DIR}/Welcome.md
    sed -i 's#notebooks\#explainer-tutorial-notebooks#<notebooks>#g' ${TEMP_DIR}/Welcome.md

    # Create an Overview doc
    sed -n '/^ *## Overview *$/,/^ *## Get Started *$/p' ${TEMP_DIR}/Welcome.md > ${TEMP_DIR}/Overview.md
    # Change the first instance of Intel to include the registered trademark symbol
    sed -i '0,/Intel/{s/Intel/Intel®/}' ${TEMP_DIR}/Overview.md
    sed -i '$d' ${TEMP_DIR}/Overview.md
    echo "*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)" >> ${TEMP_DIR}/Overview.md

    # Create an Installation doc (including requirements)
    echo "## Installation " > ${TEMP_DIR}/Install.md
    sed -n '/^ *### Requirements *$/,/^ *### Create and activate a Python3 virtual environment *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Install.md
    sed -i '$d' ${TEMP_DIR}/Install.md
    sed -i 's/### Requirements/### Software Requirements/g' ${TEMP_DIR}/Install.md
    sed -n '/^ *### Create and activate a Python3 virtual environment *$/,/^ *## Running Notebooks *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Install.md
    sed -i '$d' ${TEMP_DIR}/Install.md
    # Change the first instance of the tool name to include the registered trademark symbol
    sed -i '0,/Intel Transfer Learning Tool/{s/Intel Transfer Learning Tool/Intel® Transfer Learning Tool/}' ${TEMP_DIR}/Install.md
    echo "*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)" >> ${TEMP_DIR}/Install.md

    # Create a Legal Information doc
    echo "# Legal Information " > ${TEMP_DIR}/Legal.md
    sed -n '/#### DISCLAIMER/,$p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Legal.md
    sed -i 's/#### DISCLAIMER/## Disclaimer/g' ${TEMP_DIR}/Legal.md
    sed -i 's/#### License/## License/g' ${TEMP_DIR}/Legal.md
    sed -i 's/#### Datasets/## Datasets/g' ${TEMP_DIR}/Legal.md
    # Change the first instance of Intel to include the registered trademark symbol
    sed -i '0,/Intel/{s/Intel/Intel®/}' ${TEMP_DIR}/Legal.md
fi
