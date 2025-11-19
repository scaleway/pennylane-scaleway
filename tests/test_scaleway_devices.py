# Copyright 2025 Scaleway
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os

import pennylane as qml

# Credentials
SCW_PROJECT_ID = os.environ["SCW_PROJECT_ID"]
SCW_SECRET_KEY = os.environ["SCW_SECRET_KEY"]
SCW_BACKEND_NAME = os.getenv("SCW_BACKEND_NAME", "aer_simulation_pop_c16m128")
SCW_API_URL = os.getenv("SCW_API_URL")


# Fixtures
@pytest.fixture(scope="module")
def device_kwargs():
    """Module-scoped fixture for device keyword arguments."""
    return {
        "project_id": SCW_PROJECT_ID,
        "secret_key": SCW_SECRET_KEY,
        "url": SCW_API_URL,
        "backend": SCW_BACKEND_NAME,
    }


@pytest.mark.parametrize("device_name", ["scaleway.aer", "scaleway.aqt"])
def test_device_instantiation(device_name, device_kwargs):
    """Test basic device loading and session start/stop."""

    with qml.device(device_name, wires=2, **device_kwargs) as dev:
        assert dev.name == device_name
        assert dev.num_wires == 2
        assert dev._session_id is not None
        assert dev._platform.name == SCW_BACKEND_NAME

    # After 'with' block, session should be stopped
    assert dev._session_id is None

    # Test that calling stop() again raises an error
    with pytest.raises(RuntimeError, match="No session running"):
        dev.stop()
