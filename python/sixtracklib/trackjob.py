#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
import cobjects
from cobjects import CBuffer, CObject
from .config import SIXTRACKLIB_MODULES
import warnings

from . import stcommon as st
from . import config as stconf
from .particles import ParticlesSet
from .control import raise_error_if_status_not_success
from .control import ControllerBase, NodeControllerBase, ArgumentBase
from .buffer import Buffer, get_cbuffer_from_obj, AssignAddressItem
from .particles import ParticlesSet
from .beam_elements import Elements
from .opencl import ClController

from .stcommon import (
    string_to_encoded_ctypes_str,
    ctypes_str_to_decoded_string,
    st_TrackJobBaseNew_p,
    st_NullTrackJobBaseNew,
    st_ARCH_STATUS_SUCCESS,
    st_ARCH_STATUS_GENERAL_FAILURE,
    st_TRACK_SUCCESS,
    st_TRACK_STATUS_GENERAL_FAILURE,
    st_buffer_flags_t,
    st_arch_size_t,
    st_object_type_id_t,
    st_BUFFER_DEFAULT_CAPACITY,
    st_BUFFER_DEFAULT_DATASTORE_FLAGS,
    st_Particles_p,
    st_NullParticles,
    st_ParticlesAddr,
    st_NullParticlesAddr,
    st_track_status_t,
    st_track_job_collect_flag_t,
    st_track_job_clear_flag_t,
    st_out_buffer_flags_t,
    st_track_job_size_t,
    st_NullTrackJob,
    st_TrackJobNew_delete,
    st_TrackJobNew_track_until,
    st_buffer_size_t,
    st_NullBuffer,
    st_Buffer_is_particles_buffer,
    st_Particles_buffer_get_num_of_particle_blocks,
    st_Particles_buffer_get_particles,
    st_Particles_get_num_of_particles,
    st_OutputBuffer_required_for_tracking,
    st_OutputBuffer_requires_output_buffer,
    st_OutputBuffer_calculate_output_buffer_params,
    st_TrackJob_get_type_str,
    st_TrackJobNew_track_elem_by_elem,
    st_TrackJobNew_track_line,
    st_TrackJobNew_collect,
    st_TrackJobNew_collect_detailed,
    st_TrackJobNew_collect_particles,
    st_TrackJobNew_collect_beam_elements,
    st_TrackJobNew_collect_output,
    st_TrackJobNew_collect_debug_flag,
    st_TrackJobNew_collect_particles_addresses,
    st_TrackJobNew_enable_collect_particles,
    st_TrackJobNew_disable_collect_particles,
    st_TrackJobNew_is_collecting_particles,
    st_TrackJobNew_enable_collect_beam_elements,
    st_TrackJobNew_disable_collect_beam_elements,
    st_TrackJobNew_is_collecting_beam_elements,
    st_TrackJobNew_enable_collect_output,
    st_TrackJobNew_disable_collect_output,
    st_TrackJobNew_is_collecting_output,
    st_TrackJobNew_get_collect_flags,
    st_TrackJobNew_set_collect_flags,
    st_TrackJobNew_requires_collecting,
    st_TrackJobNew_push,
    st_TrackJobNew_push_particles,
    st_TrackJobNew_push_beam_elements,
    st_TrackJobNew_push_output,
    st_TrackJobNew_can_fetch_particle_addresses,
    st_TrackJobNew_has_particle_addresses,
    st_TrackJobNew_fetch_particle_addresses,
    st_TrackJobNew_clear_particle_addresses,
    st_TrackJobNew_clear_all_particle_addresses,
    st_TrackJobNew_get_particle_addresses,
    st_TrackJobNew_get_ptr_particle_addresses_buffer,
    st_TrackJobNew_is_in_debug_mode,
    st_TrackJobNew_enable_debug_mode,
    st_TrackJobNew_disable_debug_mode,
    st_TrackJobNew_clear,
    st_TrackJobNew_reset,
    st_TrackJobNew_reset_particle_set,
    st_TrackJobNew_reset_with_output,
    st_TrackJobNew_reset_detailed,
    st_TrackJobNew_select_particle_set,
    st_TrackJobNew_assign_output_buffer,
    st_TrackJobNew_get_arch_id,
    st_TrackJobNew_has_arch_string,
    st_TrackJobNew_get_arch_string,
    st_TrackJobNew_has_config_str,
    st_TrackJobNew_get_config_str,
    st_TrackJobNew_get_num_particle_sets,
    st_TrackJobNew_get_particle_set_indices_begin,
    st_TrackJobNew_get_particle_set_indices_end,
    st_TrackJobNew_get_particle_set_index,
    st_TrackJobNew_get_total_num_of_particles,
    st_TrackJobNew_get_min_particle_id,
    st_TrackJobNew_get_max_particle_id,
    st_TrackJobNew_get_min_element_id,
    st_TrackJobNew_get_max_element_id,
    st_TrackJobNew_get_min_initial_turn_id,
    st_TrackJobNew_get_max_initial_turn_id,
    st_TrackJobNew_get_particles_buffer,
    st_TrackJobNew_get_beam_elements_buffer,
    st_TrackJobNew_has_output_buffer,
    st_TrackJobNew_owns_output_buffer,
    st_TrackJobNew_has_elem_by_elem_output,
    st_TrackJobNew_has_beam_monitor_output,
    st_TrackJobNew_get_beam_monitor_output_buffer_offset,
    st_TrackJobNew_get_elem_by_elem_output_buffer_offset,
    st_TrackJobNew_get_num_elem_by_elem_turns,
    st_TrackJobNew_get_output_buffer,
    st_TrackJobNew_has_beam_monitors,
    st_TrackJobNew_get_num_beam_monitors,
    st_TrackJobNew_get_beam_monitor_indices_begin,
    st_TrackJobNew_get_beam_monitor_indices_end,
    st_TrackJobNew_get_beam_monitor_index,
    st_TrackJobNew_has_elem_by_elem_config,
    st_TrackJobNew_get_elem_by_elem_config,
    st_TrackJobNew_is_elem_by_elem_config_rolling,
    st_TrackJobNew_get_default_elem_by_elem_config_rolling_flag,
    st_TrackJobNew_set_default_elem_by_elem_config_rolling_flag,
    st_TrackJobNew_get_elem_by_elem_config_order,
    st_TrackJobNew_get_default_elem_by_elem_config_order,
    st_TrackJobNew_set_default_elem_by_elem_config_order,
    st_TrackJobNew_uses_controller,
    st_TrackJobNew_uses_arguments,
)

if stconf.SIXTRACKLIB_MODULES.get("opencl", False):
    from .stcommon import (
        st_OpenCL_get_available_nodes_required_str_capacity,
        st_OpenCL_get_all_nodes_required_str_capacity,
        st_OpenCL_get_available_nodes_as_string,
        st_OpenCL_get_all_nodes_as_string,
    )


class TrackJobBaseNew(object):
    def __init__(self, ptr_track_job=None, owns_ptr=True):
        self._ptr_track_job = st_NullTrackJobBaseNew
        self._owns_ptr = True
        self._last_status = st_ARCH_STATUS_SUCCESS.value
        self._last_track_status = st_TRACK_SUCCESS.value

        self._output_buffer = None
        self._particles_buffer = None
        self._beam_elements_buffer = None

        self._internal_particles_buffer = None
        self._internal_beam_elements_buffer = None
        self._internal_output_buffer = None

        self._ptr_c_particles_buffer = st_NullBuffer
        self._ptr_c_beam_elements_buffer = st_NullBuffer
        self._ptr_c_output_buffer = st_NullBuffer
        self._stored_buffers = {}

        if (
            ptr_track_job is not None
            and ptr_track_job != st_NullTrackJobBaseNew
        ):
            self._ptr_track_job = ptr_track_job
            self._owns_ptr = owns_ptr

    def __del__(self):
        if (
            self._ptr_track_job is not None
            and self._ptr_track_job != st_NullTrackJobBaseNew
            and self._owns_ptr
        ):
            st_TrackJobNew_delete(self._ptr_track_job)
            self._ptr_track_job = st_NullTrackJobBaseNew
            self._owns_ptr = False

    def _reset_detailed(
        self,
        beam_elements_buffer,
        particles_buffer,
        particle_set_index=0,
        until_turn_elem_by_elem=0,
        output_buffer=None,
    ):
        if (
            self._ptr_track_job is None
            or self._ptr_track_job == st_NullTrackJobBaseNew
        ):
            raise ValueError(
                "TrackJob has to be initialized before " + "calling reset"
            )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        _particles_buffer = None
        _internal_particles_buffer = None
        _ptr_c_particles_buffer = st_NullBuffer

        if particles_buffer is not None:
            if isinstance(particles_buffer, CBuffer):
                _particles_buffer = particles_buffer
                _internal_particles_buffer = Buffer(cbuffer=particles_buffer)
                _ptr_c_particles_buffer = _internal_particles_buffer.pointer
            elif isinstance(particles_buffer, CObject):
                _particles_buffer = particles_buffer._buffer
                _internal_particles_buffer = Buffer(cbuffer=_particles_buffer)
                _ptr_c_particles_buffer = _internal_particles_buffer.pointer
            elif isinstance(particles_buffer, ParticlesSet):
                _particles_buffer = particles_buffer.cbuffer
                _internal_particles_buffer = Buffer(cbuffer=_particles_buffer)
                _ptr_c_particles_buffer = _internal_particles_buffer.pointer
            elif isinstance(particles_buffer, Buffer):
                _internal_particles_buffer = particles_buffer
                _ptr_c_particles_buffer = particles_buffer.pointer

        if (
            _internal_particles_buffer is None
            or _ptr_c_particles_buffer == st_NullBuffer
        ):
            raise ValueError("Issues with input particles buffer")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        slot_size = _internal_particles_buffer.slot_size
        _slot_size = st_buffer_size_t(slot_size)
        particles = st_NullParticles

        if (
            st_Particles_buffer_get_num_of_particle_blocks(
                _ptr_c_particles_buffer
            )
            > particle_set_index
        ):
            _pset_index = st_buffer_size_t(particle_set_index)
            particles = st_Particles_buffer_get_particles(
                _ptr_c_particles_buffer, _pset_index
            )
        else:
            raise ValueError(
                "Input particle buffer has {0} blocks, "
                + "inconsistent with particle set index {1}".format(
                    st_Particles_buffer_get_num_of_particle_blocks(
                        _ptr_c_particles_buffer
                    ),
                    particle_set_index,
                )
            )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        _beam_elem_buffer = None
        _internal_beam_elements_buffer = None
        _ptr_c_beam_elements_buffer = st_NullBuffer

        if beam_elements_buffer is not None:
            if isinstance(beam_elements_buffer, CBuffer):
                _beam_elements_buffer = beam_elements_buffer
                _internal_beam_elements_buffer = Buffer(
                    cbuffer=beam_elements_buffer
                )
                _ptr_c_beam_elements_buffer = (
                    _internal_beam_elements_buffer.pointer
                )
            elif isinstance(beam_elements_buffer, CObject):
                _beam_elements_buffer = beam_elements_buffer._buffer
                _internal_beam_elements_buffer = Buffer(
                    cbuffer=_beam_elements_buffer
                )
                _ptr_c_beam_elements_buffer = (
                    _internal_beam_elements_buffer.pointer
                )
            elif isinstance(beam_elements_buffer, Elements):
                _beam_elements_buffer = beam_elements_buffer.cbuffer
                _internal_beam_elements_buffer = Buffer(
                    cbuffer=_beam_elements_buffer
                )
                _ptr_c_beam_elements_buffer = (
                    _internal_beam_elements_buffer.pointer
                )
            elif isinstance(beam_elements_buffer, Buffer):
                _internal_beam_elements_buffer = beam_elements_buffer
                _ptr_c_beam_elements_buffer = beam_elements_buffer.pointer

        if (
            _internal_beam_elements_buffer is None
            or _ptr_c_beam_elements_buffer == st_NullBuffer
        ):
            raise ValueError("Issues with input beam elements buffer")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        _output_buffer = None
        _internal_output_buffer = None
        _ptr_c_output_buffer = st_NullBuffer

        if output_buffer is not None:
            if isinstance(output_buffer, CBuffer):
                _output_buffer = output_buffer
            elif isinstance(output_buffer, CObject):
                _output_buffer = output_buffer._buffer
            elif isinstance(output_buffer, ParticlesSet):
                _output_buffer = output_buffer.cbuffer
            elif isinstance(output_buffer, Buffer):
                _internal_output_buffer = output_buffer
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        _until_turn_elem_by_elem = st_buffer_size_t(until_turn_elem_by_elem)

        out_buffer_flags = st_OutputBuffer_required_for_tracking(
            particles, _ptr_c_beam_elements_buffer, _until_turn_elem_by_elem
        )

        _out_buffer_flags = st_out_buffer_flags_t(out_buffer_flags)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        needs_output_buffer = st_OutputBuffer_requires_output_buffer(
            _out_buffer_flags
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if needs_output_buffer:
            num_objects = st_buffer_size_t(0)
            num_slots = st_buffer_size_t(0)
            num_dataptrs = st_buffer_size_t(0)
            num_garbage = st_buffer_size_t(0)

            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                _ptr_c_beam_elements_buffer,
                particles,
                _until_turn_elem_by_elem,
                ct.byref(num_objects),
                ct.byref(num_slots),
                ct.byref(num_dataptrs),
                ct.byref(num_garbage),
                _slot_size,
            )

            if ret != 0:
                raise RuntimeError("Error pre-calculating out buffer params")

            if (
                num_objects.value > 0
                and num_slots.value > 0
                and num_dataptrs.value > 0
                and num_garbage.value >= 0
            ):
                if _output_buffer is None and _internal_output_buffer is None:
                    _output_buffer = CBuffer(
                        max_slots=num_slots.value,
                        max_objects=num_objects.value,
                        max_pointers=num_dataptrs.value,
                        max_garbage=num_garbage.value,
                    )
                    _internal_output_buffer = Buffer(cbuffer=_output_buffer)
                elif _output_buffer is not None:
                    _output_buffer.reallocate(
                        max_slots=num_slots.value,
                        max_objects=num_objects.value,
                        max_pointers=num_dataptrs.value,
                        max_garbage=num_garbage.value,
                    )
                    _internal_output_buffer = Buffer(cbuffer=_output_buffer)
                elif _internal_output_buffer is not None:
                    _internal_output_buffer.reserve(
                        num_objects.value,
                        num_slots.value,
                        num_dataptrs.value,
                        num_garbage.value,
                    )
                else:
                    raise ValueError("No valid output buffer available")

                if _internal_output_buffer is None:
                    raise ValueError("Could not provide output buffer")

                _ptr_c_output_buffer = _internal_output_buffer.pointer
        elif _output_buffer is not None:
            _internal_output_buffer = Buffer(cbuffer=_output_buffer)
            _ptr_c_output_buffer = _internal_output_buffer.pointer

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if (
            needs_output_buffer
            and (
                _ptr_c_output_buffer == st_NullBuffer
                or _internal_output_buffer is None
            )
        ) or (
            (_output_buffer is not None or _internal_output_buffer is not None)
            and _ptr_c_output_buffer == st_NullBuffer
        ):
            raise RuntimeError("Unable to provide output buffer")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._last_status = st_TrackJobNew_reset_detailed(
            self._ptr_track_job,
            _ptr_c_particles_buffer,
            st_buffer_size_t(1),
            ct.byref(_pset_index),
            _ptr_c_beam_elements_buffer,
            _ptr_c_output_buffer,
            _until_turn_elem_by_elem,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if self._last_status == st_ARCH_STATUS_SUCCESS.value:
            self._ptr_c_particles_buffer = st_NullBuffer
            self._ptr_c_beam_elements_buffer = st_NullBuffer
            self._ptr_c_output_buffer = st_NullBuffer

            self._ptr_c_particles_buffer = _ptr_c_particles_buffer
            self._ptr_c_beam_elements_buffer = _ptr_c_beam_elements_buffer
            self._ptr_c_output_buffer = _ptr_c_output_buffer

            del self._internal_particles_buffer
            del self._internal_beam_elements_buffer
            del self._internal_output_buffer

            self._internal_beam_elements_buffer = (
                _internal_beam_elements_buffer
            )
            self._internal_particles_buffer = _internal_particles_buffer
            self._internal_output_buffer = _internal_output_buffer

            del self._particles_buffer
            del self._beam_elements_buffer
            del self._output_buffer

            self._particles_buffer = _particles_buffer
            self._beam_elements_buffer = beam_elements_buffer
            self._output_buffer = _output_buffer
        else:
            raise RuntimeError("Error while resetting the trackjob")

    @property
    def pointer(self):
        return self._ptr_track_job

    @property
    def last_status(self):
        return self._last_status

    @property
    def last_status_success(self):
        return self._last_status == st_ARCH_STATUS_SUCCESS.value

    @property
    def last_track_status(self):
        return self._last_track_status

    @property
    def last_track_status_success(self):
        return self._last_track_status == st_TRACK_SUCCESS.value

    @property
    def arch_id(self):
        return st_TrackJobNew_get_arch_id(self._ptr_track_job)

    @property
    def arch_str(self):
        arch_str = None
        if st_TrackJobNew_has_arch_string(self._ptr_track_job):
            arch_str = ctypes_str_to_decoded_string(
                st_TrackJobNew_get_arch_string(self._ptr_track_job)
            )
        return arch_str

    @property
    def uses_controller(self):
        return st_TrackJobNew_uses_controller(self._ptr_track_job)

    @property
    def uses_arguments(self):
        return st_TrackJobNew_uses_arguments(self._ptr_track_job)

    @property
    def is_in_debug_mode(self):
        return st_TrackJobNew_is_in_debug_mode(self._ptr_track_job)

    @property
    def requires_collecting(self):
        return st_TrackJobNew_requires_collecting(self._ptr_track_job)

    @property
    def is_collecting_particles(self):
        return st_TrackJobNew_is_collecting_particles(self._ptr_track_job)

    @property
    def is_collecting_beam_elements(self):
        return st_TrackJobNew_is_collecting_beam_elements(self._ptr_track_job)

    @property
    def is_collecting_output(self):
        return st_TrackJobNew_is_collecting_output(self._ptr_track_job)

    @property
    def collecting_flags(self):
        return st_TrackJobNew_get_collect_flags(self._ptr_track_job)

    @property
    def can_fetch_particle_addresses(self):
        return st_TrackJobNew_can_fetch_particle_addresses(self._ptr_track_job)

    @property
    def has_particle_addresses(self):
        return st_TrackJobNew_has_particle_addresses(self._ptr_track_job)

    @property
    def num_particle_sets(self):
        return st_TrackJobNew_get_num_particle_sets(self._ptr_track_job)

    @property
    def total_num_particles(self):
        return st_TrackJobNew_get_total_num_of_particles(self._ptr_track_job)

    @property
    def has_config_str(self):
        return st_TrackJobNew_has_config_str(self._ptr_track_job)

    @property
    def has_output_buffer(self):
        return st_TrackJobNew_has_output_buffer(self._ptr_track_job)

    @property
    def owns_output_buffer(self):
        return st_TrackJobNew_owns_output_buffer(self._ptr_track_job)

    @property
    def has_elem_by_elem_output(self):
        return st_TrackJobNew_has_elem_by_elem_output(self._ptr_track_job)

    @property
    def has_beam_monitors(self):
        return st_TrackJobNew_has_beam_monitors(self._ptr_track_job)

    @property
    def num_beam_monitors(self):
        return st_TrackJobNew_get_num_beam_monitors(self._ptr_track_job)

    @property
    def has_beam_monitor_output(self):
        return st_TrackJobNew_has_beam_monitor_output(self._ptr_track_job)

    @property
    def beam_monitor_output_buffer_offset(self):
        return st_TrackJobNew_get_beam_monitor_output_buffer_offset(
            self._ptr_track_job
        )

    @property
    def elem_by_elem_output_buffer_offset(self):
        return st_TrackJobNew_get_elem_by_elem_output_buffer_offset(
            self._ptr_track_job
        )

    @property
    def num_elem_by_elem_turn(self):
        return st_TrackJobNew_get_elem_by_elem_output_buffer_offset(
            self._ptr_track_job
        )

    @property
    def has_elem_by_elem_config(self):
        return st_TrackJobNew_has_elem_by_elem_config(self._ptr_track_job)

    @property
    def output_buffer(self):
        return self._output_buffer

    @property
    def output(self):
        return ParticlesSet(self._output_buffer)

    @property
    def particles_buffer(self):
        return self._particles_buffer

    @property
    def beam_elements_buffer(self):
        return self._beam_elements_buffer

    # -------------------------------------------------------------------------

    def track_until(self, until_turn):
        self._last_track_status = st_TrackJobNew_track_until(
            self._ptr_track_job, st_buffer_size_t(until_turn)
        )
        if self._last_track_status != st_TRACK_SUCCESS.value:
            raise RuntimeError(
                "Error occuriung during track_until(); "
                "track status:{0}".format(self._last_track_status)
            )
        return self

    def track_elem_by_elem(self, until_turn):
        self._last_track_status = st_TrackJobNew_track_elem_by_elem(
            self._ptr_track_job, st_buffer_size_t(until_turn)
        )
        if self._last_track_status != st_TRACK_SUCCESS.value:
            raise RuntimeError(
                "Error occuriung during track_elem_by_elem(); "
                "track status:{0}".format(self._last_track_status)
            )
        return self

    def track_line(self, be_begin_idx, be_end_idx, finish_turn=False):
        self._last_track_status = st_TrackJobNew_track_line(
            self._ptr_track_job,
            st_buffer_size_t(be_begin_idx),
            st_buffer_size_t(be_end_idx),
            ct.c_bool(finish_turn),
        )
        if self._last_track_status != st_TRACK_SUCCESS.value:
            raise RuntimeError(
                "Error occuriung during track_line(); "
                "track status:{0}".format(self._last_track_status)
            )
        return self

    # -------------------------------------------------------------------------

    def collect(self, collect_flags=None):
        if collect_flags is None:
            collect_flags = st_TrackJobNew_get_collect_flags(
                self._ptr_track_job
            )
        if collect_flags > 0:
            collected_flags = st_TrackJobNew_collect(self._ptr_track_job)
        return self

    def collectParticles(self):
        warnings.warn(
            "collectParticles() is depreciated;"
            + "use collect_particles instead",
            DeprecationWarning,
        )
        self.collect_particles(self)

    def collect_particles(self):
        assert self._ptr_c_particles_buffer != st_NullBuffer
        assert self._internal_particles_buffer.pointer != st_NullBuffer

        self._last_status = st_TrackJobNew_collect_particles(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful particles collection op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    def collectBeamElements(self):
        warnings.warn(
            "collectBeamElements() is depreciated; "
            + "use collect_beam_elements() instead",
            DeprecationWarning,
        )
        self.collect_beam_elements()

    def collect_beam_elements(self):
        self._last_status = st_TrackJobNew_collect_beam_elements(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful beam-elements collection op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    def collectOutput(self):
        warnings.warn(
            "collectOutput() is depreciated; " + "use collect_output instead",
            DeprecationWarning,
        )
        self.collect_output()

    def collect_output(self):
        self._last_status = st_TrackJobNew_collect_output(self._ptr_track_job)
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful output collection op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    def collectParticlesAddresses(self):
        warnings.warn(
            "collectParticleAddresses() is depreciated;"
            + " use collect_particle_addresses() instead",
            DeprecationWarning,
        )
        self.collect_particle_addresses()

    def collect_particle_addresses(self):
        self._last_status = st_TrackJobNew_collect_particles_addresses(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful particles addresses collection op; "
            + "status:{0}".format(self._last_status),
        )
        return self

    def collectDebugFlag(self):
        warnings.warn(
            "collectDebugFlag() is depreciated; "
            + "use collect_debug_flag() instead",
            DeprecationWarning,
        )
        self.collect_debug_flag()

    def collect_debug_flag(self):
        self._last_status = st_TrackJobNew_collect_debug_flag(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful debug flag collection op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    # -------------------------------------------------------------------------

    def push_particles(self):
        self._last_status = st_TrackJobNew_push_particles(self._ptr_track_job)
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful particles push op; status: {0}".format(
                self._last_status
            ),
        )
        return self

    def push_beam_elements(self):
        self._last_status = st_TrackJobNew_push_beam_elements(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful beam elements push op; status: {0}".format(
                self._last_status
            ),
        )
        return self

    def _push_output(self):
        self._last_status = st_TrackJobNew_push_output(self._ptr_track_job)
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful output push op; status: {0}".format(
                self._last_status
            ),
        )
        return self

    # -------------------------------------------------------------------------

    def enable_debug_mode(self):
        self._last_status = st_TrackJobNew_enable_debug_mode(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful enable debug mode op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    def disable_debug_mode(self):
        self._last_status = st_TrackJobNew_disable_debug_mode(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful disable debug mode op; status:{0}".format(
                self._last_status
            ),
        )
        return self

    # -------------------------------------------------------------------------

    def fetch_particle_addresses(self):
        self._last_status = st.st_TrackJobNew_fetch_particle_addresses(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful fetching particle addresses op; "
            + "status:{0}".format(self._last_status),
        )
        return self

    def clear_particle_addresses(self, particle_set_index=0):
        self._last_status = st.st_TrackJobNew_clear_particle_addresses(
            self._ptr_track_job, st_buffer_size_t(particle_set_index)
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful clearing of particle addresses op; "
            + "particle_set_index={0}, status:{1}".format(
                particle_set_index, self._last_status
            ),
        )
        return self

    def clear_all_particle_addresses(self):
        self._last_status = st.st_TrackJobNew_clear_all_particle_addresses(
            self._ptr_track_job
        )
        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful clearing all particle addresses op; "
            + "status:{0}".format(self._last_status),
        )
        return self

    def get_particle_addresses(self, particle_set_index=0):
        return st.st_TrackJobNew_get_particle_addresses(
            self._ptr_track_job, st_buffer_size_t(particle_set_index)
        )


class TrackJob(object):
    @staticmethod
    def num_all_nodes(arch_str=None):
        if (
            not (arch_str is None)
            and arch_str == "opencl"
            and stconf.SIXTRACKLIB_MODULES.get("opencl", False)
        ):
            return st.st_OpenCL_get_num_all_nodes()
        else:
            return 0

    @staticmethod
    def num_available_nodes(arch_str=None, env_var_name=None, filter_str=None):
        if (
            not (arch_str is None)
            and arch_str == "opencl"
            and stconf.SIXTRACKLIB_MODULES.get("opencl", False)
        ):
            return st.st_OpenCL_num_available_nodes_detailed(
                string_to_encoded_ctypes_str(filter_str),
                string_to_encoded_ctypes_str(env_var_name),
            )
        else:
            return 0

    @staticmethod
    def enabled_archs():
        enabled_archs = [
            arch_str for arch_str, flag in SIXTRACKLIB_MODULES.items() if flag
        ]
        if "cpu" not in SIXTRACKLIB_MODULES:
            enabled_archs.append("cpu")
        return enabled_archs

    @staticmethod
    def print_nodes(arch_str, all=False, filter_str=None, env_var_name=None):
        arch_str = arch_str.strip().lower()
        _filter_str = string_to_encoded_ctypes_str(filter_str)
        _env_var_name = string_to_encoded_ctypes_str(env_var_name)

        if stconf.SIXTRACKLIB_MODULES.get(arch_str, False):
            if arch_str == "opencl":
                if not all:
                    cap = st_OpenCL_get_available_nodes_required_str_capacity(
                        _filter_str, _env_var_name
                    )
                else:
                    cap = st_OpenCL_get_all_nodes_required_str_capacity()

                if cap > 0:
                    _nodes_str = ct.create_string_buffer(cap)
                    cap = st_arch_size_t(cap)

                    if not all:
                        _status = st_OpenCL_get_available_nodes_as_string(
                            _nodes_str, cap, _filter_str, _env_var_name
                        )
                    else:
                        _status = st_OpenCL_get_all_nodes_as_string(
                            _nodes_str, cap
                        )

                    if _status == st_ARCH_STATUS_SUCCESS.value:
                        print(ctypes_str_to_decoded_string(_nodes_str))
                    else:
                        raise RuntimeError("unable to print opencl nodes")
            else:
                print(f"nodes not available for architecture {arch_str}")
        else:
            print(f"architecture {arc_str} is not enabled/known")

    def __init__(
        self,
        beam_elements_buffer,
        particles_buffer,
        until_turn_elem_by_elem=0,
        arch="cpu",
        device_id=None,
        device=None,
        output_buffer=None,
        config_str=None,
    ):
        self.ptr_st_track_job = st.st_NullTrackJob
        self._particles_buffer = None
        self._ptr_c_particles_buffer = st.st_NullBuffer
        self._beam_elements_buffer = None
        self._ptr_c_beam_elements_buffer = st.st_NullBuffer
        self._output_buffer = None
        self._ptr_c_output_buffer = st.st_NullBuffer
        self._stored_buffers = {}
        self._ext_stored_st_buffers = {}
        self._last_status = st_ARCH_STATUS_SUCCESS.value
        self._last_track_status = st_TRACK_SUCCESS.value

        base_addr_t = ct.POINTER(ct.c_ubyte)
        success = False

        if particles_buffer is not None:
            particles_buffer = get_cbuffer_from_obj(particles_buffer)
            self._particles_buffer = particles_buffer
            self._ptr_c_particles_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                particles_buffer
            )
            if self._ptr_c_particles_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input particles buffer")

        if beam_elements_buffer is not None:
            beam_elements_buffer = get_cbuffer_from_obj(beam_elements_buffer)
            self._beam_elements_buffer = beam_elements_buffer
            self._ptr_c_beam_elements_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                beam_elements_buffer
            )
            if self._ptr_c_beam_elements_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input beam elements buffer")

        particles = st.st_Particles_buffer_get_particles(
            self._ptr_c_particles_buffer, 0
        )

        if particles == st.st_NullParticles:
            raise ValueError("Required particle sets not available")

        until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)
        out_buffer_flags = st.st_OutputBuffer_required_for_tracking(
            particles,
            self._ptr_c_beam_elements_buffer,
            until_turn_elem_by_elem,
        )
        needs_output_buffer = st.st_OutputBuffer_requires_output_buffer(
            ct.c_int32(out_buffer_flags)
        )

        if needs_output_buffer:
            num_objects = ct.c_uint64(0)
            num_slots = ct.c_uint64(0)
            num_dataptrs = ct.c_uint64(0)
            num_garbage = ct.c_uint64(0)
            slot_size = st.st_Buffer_get_slot_size(
                self._ptr_c_particles_buffer
            )

            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                self._ptr_c_beam_elements_buffer,
                particles,
                until_turn_elem_by_elem,
                ct.byref(num_objects),
                ct.byref(num_slots),
                ct.byref(num_dataptrs),
                ct.byref(num_garbage),
                slot_size,
            )

            if ret == 0:
                if (
                    num_objects.value > 0
                    and num_slots.value > 0
                    and num_dataptrs.value > 0
                    and num_garbage.value >= 0
                ):
                    if output_buffer is None:
                        output_buffer = CBuffer(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value,
                        )
                    else:
                        output_buffer.reallocate(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value,
                        )

                    if output_buffer is None:
                        raise ValueError("Could not provide output buffer")

                self._output_buffer = output_buffer
                self._ptr_c_output_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                    output_buffer
                )
                if self._ptr_c_output_buffer == st.st_NullBuffer:
                    raise ValueError("Unable to map (optional) output buffer")
            else:
                raise ValueError("Error pre-calculating out buffer params")
        elif output_buffer is not None:
            self._output_buffer = output_buffer
            self._ptr_c_output_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                self._output_buffer
            )
            if self._ptr_c_output_buffer == st.st_NullBuffer:
                raise ValueError("Unable to map (optional) output buffer")

        assert (
            needs_output_buffer
            and self._ptr_c_output_buffer != st.st_NullBuffer
        ) or (not needs_output_buffer)

        if device is not None:
            arch, device_id = device.split(":")

        arch = arch.strip().lower()
        if not (
            stconf.SIXTRACKLIB_MODULES.get(arch, False) is not False
            or arch == "cpu"
        ):
            raise ValueError("Unknown architecture {0}".format(arch,))

        if device_id is not None:
            if config_str is None:
                config_str = device_id
            else:
                config_str = device_id + ";" + config_str
        else:
            config_str = ""

        _arch = string_to_encoded_ctypes_str(arch)
        _config_str = string_to_encoded_ctypes_str(config_str)

        self.ptr_st_track_job = st.st_TrackJob_new_with_output(
            _arch,
            self._ptr_c_particles_buffer,
            self._ptr_c_beam_elements_buffer,
            self._ptr_c_output_buffer,
            until_turn_elem_by_elem,
            _config_str,
        )

        if self.ptr_st_track_job == st.st_NullTrackJob:
            raise ValueError("unable to construct TrackJob from arguments")

    def __del__(self):
        if self.ptr_st_track_job != st.st_NullTrackJob:
            job_owns_output_buffer = st.st_TrackJob_owns_output_buffer(
                self.ptr_st_track_job
            )

            st.st_TrackJob_delete(self.ptr_st_track_job)
            self.ptr_st_track_job = st.st_NullTrackJob

            if (
                job_owns_output_buffer
                and self._ptr_c_output_buffer != st.st_NullBuffer
            ):
                self._ptr_c_output_buffer = st.st_NullBuffer

        if self._ptr_c_particles_buffer != st.st_NullBuffer:
            st.st_Buffer_delete(self._ptr_c_particles_buffer)
            self._ptr_c_particles_buffer = st.st_NullBuffer

        if self._ptr_c_beam_elements_buffer != st.st_NullBuffer:
            st.st_Buffer_delete(self._ptr_c_beam_elements_buffer)
            self._ptr_c_beam_elements_buffer = st.st_NullBuffer

        if self._ptr_c_output_buffer != st.st_NullBuffer:
            st.st_Buffer_delete(self._ptr_c_output_buffer)
            self._ptr_c_output_buffer = st.st_NullBuffer

        if len(self._ext_stored_st_buffers) > 0:
            self._ext_stored_st_buffers = None

    @property
    def last_status(self):
        return self._last_status

    @property
    def last_track_status(self):
        return self._last_track_status

    @property
    def output_buffer(self):
        return self._output_buffer

    @property
    def output(self):
        return ParticlesSet(self._output_buffer)

    @property
    def particles_buffer(self):
        return self._particles_buffer

    @property
    def beam_elements_buffer(self):
        return self._beam_elements_buffer

    def track(self, until_turn):
        warnings.warn(
            "track(until_turn) is depreciated; "
            + "use track_until(until_turn) instead",
            DeprecationWarning,
        )
        return self.track_until(until_turn)

    def track_until(self, until_turn):
        self._last_track_status = st.st_TrackJob_track_until(
            self.ptr_st_track_job, ct.c_uint64(until_turn)
        )
        if self._last_track_status != st.st_TRACK_SUCCESS.value:
            raise RuntimeError(
                f"Error during performing track_until({until_turn})"
            )
        return self

    def track_elem_by_elem(self, until_turn):
        self._last_track_status = st.st_TrackJob_track_elem_by_elem(
            self.ptr_st_track_job, ct.c_uint64(until_turn)
        )
        if self._last_track_status != st.st_TRACK_SUCCESS.value:
            raise RuntimeError(
                f"Error during performing track_elem_by_elem({until_turn})"
            )
        return self

    def track_line(self, begin_idx, end_idx, finish_turn=False):
        self._last_track_status = st.st_TrackJob_track_line(
            self.ptr_st_track_job,
            ct.c_uint64(begin_idx),
            ct.c_uint64(end_idx),
            ct.c_bool(finish_turn),
        )
        if self._last_track_status != st.st_TRACK_SUCCESS.value:
            raise RuntimeError(
                f"Error during performing track_line({until_turn})"
            )
        return self

    def collect(self):
        st.st_TrackJob_collect(self.ptr_st_track_job)
        return

    def collect_particles(self):
        st.st_TrackJob_collect_particles(self.ptr_st_track_job)
        return

    def collect_beam_elements(self):
        st.st_TrackJob_collect_beam_elements(self.ptr_st_track_job)
        return

    def collect_output(self):
        st.st_TrackJob_collect_output(self.ptr_st_track_job)

    @property
    def requires_collecting(self):
        return st.st_TrackJob_requires_collecting(self.ptr_st_track_job)

    def push_particles(self):
        st.st_TrackJob_push_particles(self.ptr_st_track_job)
        return

    def push_beam_elements(self):
        st.st_TrackJob_push_beam_elements(self.ptr_st_track_job)
        return

    def _push_output(self):
        st.st_TrackJob_push_output(self.ptr_st_track_job)

    @property
    def can_fetch_particle_addresses(self):
        return st.st_TrackJob_can_fetch_particles_addr(self.ptr_st_track_job)

    @property
    def has_particle_addresses(self):
        return st.st_TrackJob_has_particles_addr(self.ptr_st_track_job)

    def type(self):
        return st.st_TrackJob_get_type_id(self.ptr_st_track_job)

    def type_str(self):
        return ctypes_str_to_decoded_string(
            st_TrackJob_get_type_str(self.ptr_st_track_job)
        )

    @property
    def arch_id(self):
        return st.st_TrackJob_get_type_id(self.ptr_st_track_job)

    @property
    def arch_str(self):
        return ctypes_str_to_decoded_string(
            st_TrackJob_get_type_str(self.ptr_st_track_job)
        )

    @property
    def num_beam_monitors(self):
        return st.st_TrackJob_get_num_beam_monitors(self.ptr_st_track_job)

    @property
    def has_elem_by_elem_output(self):
        return st.st_TrackJob_has_elem_by_elem_output(self.ptr_st_track_job)

    @property
    def has_beam_monitor_output(self):
        return st.st_TrackJob_has_beam_monitor_output(self.ptr_st_track_job)

    @property
    def elem_by_elem_output_offset(self):
        return st.st_TrackJob_get_elem_by_elem_output_buffer_offset(
            self.ptr_st_track_job
        )

    @property
    def beam_monitor_output_offset(self):
        return st.st_TrackJob_get_beam_monitor_output_buffer_offset(
            self.ptr_st_track_job
        )

    @property
    def has_output_buffer(self):
        return st.st_TrackJob_has_output_buffer(
            self.ptr_st_track_job
        ) and bool(self._output_buffer is not None)

    @property
    def allows_add_program(self):
        return self.arch_str() == "opencl"

    @property
    def allows_enable_kernel(self):
        return self.arch_str() == "opencl"

    @property
    def has_stored_buffers(self):
        return st.st_TrackJob_has_stored_buffers(self.ptr_st_track_job)

    @property
    def num_stored_buffers(self):
        return st.st_TrackJob_num_stored_buffers(self.ptr_st_track_job)

    @property
    def min_stored_buffer_id(self):
        return st.st_TrackJob_min_stored_buffer_id(self.ptr_st_track_job)

    @property
    def max_stored_buffer_id(self):
        return st.st_TrackJob_max_stored_buffer_id(self.ptr_st_track_job)

    @property
    def controller(self):
        if self.arch_str == "opencl" and SIXTRACKLIB_MODULES.get(
            self.arch_str, False
        ):
            if self.ptr_st_track_job == st_NullTrackJob:
                raise RuntimeError("TrackJob is not initialized yet")
            return ClController(
                ext_ptr_ctrl=st.st_TrackJobCl_get_context(
                    self.ptr_st_track_job
                ),
                owns_ptr=False,
            )
        else:
            raise RuntimeError(
                "TrackJob has no controller for this architecture"
            )
        return None

    def add_program(self, path_to_program_file, compile_options):
        if not self.allows_add_program:
            raise RuntimeError("Can not add a program to this TrackJob")
        success = False
        program_id = st.st_ARCH_ILLEGAL_PROGRAM_ID.value

        if self.arch_str == "opencl" and SIXTRACKLIB_MODULES.get(
            self.arch_str, False
        ):
            _controller = st.st_TrackJobCl_get_context(self.ptr_st_track_job)
            program_id = st.st_ClContextBase_add_program_file(
                _controller,
                string_to_encoded_ctypes_str(path_to_program_file),
                string_to_encoded_ctypes_str(compile_options),
            )

            if program_id != st.st_ARCH_ILLEGAL_PROGRAM_ID.value:
                success = st.st_ClContextBase_compile_program(
                    _controller, ct.c_uint32(program_id)
                )
                if not (success):
                    program_id = st.st_ARCH_ILLEGAL_PROGRAM_ID.value
            else:
                raise RuntimeError("Unable to load program")

        if program_id == st.st_ARCH_ILLEGAL_PROGRAM_ID.value:
            raise RuntimeError("Unable to compile program")
        return program_id

    def enable_kernel(self, program_id, kernel_name):
        if not self.allows_enable_kernel:
            raise RuntimeError("Can not enable a kernel at this TrackJob")

        if program_id == st.st_ARCH_ILLEGAL_PROGRAM_ID.value:
            raise ValueError("Illegal program_id provided")

        kernel_id = st.st_ARCH_ILLEGAL_KERNEL_ID.value

        if self.arch_str == "opencl" and SIXTRACKLIB_MODULES.get(
            self.arch_str, False
        ):
            _controller = st.st_TrackJobCl_get_context(self.ptr_st_track_job)
            kernel_id = st.st_ClContextBase_enable_kernel(
                _controller,
                string_to_encoded_ctypes_str(kernel_name),
                program_id,
            )

        if kernel_id == st.st_ARCH_ILLEGAL_KERNEL_ID.value:
            raise RuntimeError("Unable to enable kernel")
        return kernel_id

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

    @property
    def total_num_assign_items(self):
        return st.st_TrackJob_total_num_assign_items(self.ptr_st_track_job)

    def ptr_assign_address_item(
        self,
        item=None,
        dest_elem_type_id=None,
        dest_buffer_id=None,
        dest_elem_index=None,
        dest_pointer_offset=None,
        src_elem_type_id=None,
        src_buffer_id=None,
        src_elem_index=None,
        src_pointer_offset=None,
        index=None,
    ):
        _ptr_found = st.st_NullAssignAddressItem
        if not (item is None):
            if isinstance(item, AssignAddressItem):
                # TODO: Figure out a way to do this without relying on
                #      internal API for cobjects
                _ptr_found = st.st_TrackJob_ptr_assign_address_item(
                    self.ptr_st_track_job,
                    ct.cast(item._get_address(), st.st_AssignAddressItem_p),
                )
            elif item != st.st_NullAssignAddressItem:
                _ptr_found = st.st_TrackJob_ptr_assign_address_item(
                    self.ptr_st_track_job, item
                )
        elif not (dest_buffer_id is None) and not (src_buffer_id is None):
            if not (index is None) and index >= 0:
                _ptr_found = st.st_TrackJob_ptr_assign_address_item_by_index(
                    self.ptr_st_track_job,
                    st_buffer_size_t(dest_buffer_id),
                    st_buffer_size_t(src_buffer_id),
                    st_buffer_size_t(index),
                )
            elif (
                not (dest_elem_type_id is None)
                and not (dest_elem_index is None)
                and not (dest_pointer_offset is None)
                and not (src_elem_type_id is None)
                and not (src_elem_index is None)
                and not (src_pointer_offset is None)
            ):
                _ptr_found = st.st_TrackJob_ptr_assign_address_item_detailed(
                    self.ptr_st_track_job,
                    st_object_type_id_t(dest_elem_type_id),
                    st_buffer_size_t(dest_buffer_id),
                    st_buffer_size_t(dest_elem_index),
                    st_buffer_size_t(dest_pointer_offset),
                    st_object_type_id_t(src_elem_type_id),
                    st_buffer_size_t(src_buffer_id),
                    st_buffer_size_t(src_elem_index),
                    st_buffer_size_t(src_pointer_offset),
                )
        return _ptr_found

    def has_assign_items(self, dest_buffer_id, src_buffer_id):
        return st.st_TrackJob_has_assign_items(
            self.ptr_st_track_job,
            st_buffer_size_t(dest_buffer_id),
            st_buffer_size_t(src_buffer_id),
        )

    def num_assign_items(self, dest_buffer_id, src_buffer_id):
        return st.st_TrackJob_num_assign_items(
            self.ptr_st_track_job,
            st_buffer_size_t(dest_buffer_id),
            st_buffer_size_t(src_buffer_id),
        )

    def has_assign_item(
        self,
        item=None,
        dest_buffer_id=None,
        src_buffer_id=None,
        index=None,
        dest_elem_type_id=None,
        dest_elem_index=None,
        dest_pointer_offset=None,
        src_elem_type_id=None,
        src_elem_index=None,
        src_pointer_offset=None,
    ):
        has_item = False
        if not (item is None):
            if isinstance(item, AssignAddressItem):
                # TODO: Figure out a way to do this without relying on
                #      internal API for cobjects
                has_item = st.st_TrackJob_has_assign_address_item(
                    self.ptr_st_track_job,
                    ct.cast(item._get_address(), st.st_AssignAddressItem_p),
                )
            elif item != st.st_NullAssignAddressItem:
                has_item = st.st_TrackJob_has_assign_address_item(
                    self.ptr_st_track_job, item
                )
        elif not (dest_buffer_id is None) and not (src_buffer_id is None):
            if not (index is None) and index >= 0:
                has_item = st.st_TrackJob_has_assign_item_by_index(
                    self.ptr_st_track_job,
                    st_buffer_size_t(dest_buffer_id),
                    st_buffer_size_t(src_buffer_id),
                    st_buffer_size_t(index),
                )
            elif (
                not (dest_elem_type_id is None)
                and not (dest_elem_index is None)
                and not (dest_pointer_offset is None)
                and not (src_elem_type_id is None)
                and not (src_elem_index is None)
                and not (src_pointer_offset is None)
            ):
                has_item = st.st_TrackJob_has_assign_address_item_detailed(
                    self.ptr_st_track_job,
                    st_object_type_id_t(dest_elem_type_id),
                    st_buffer_size_t(dest_buffer_id),
                    st_buffer_size_t(dest_elem_index),
                    st_buffer_size_t(dest_pointer_offset),
                    st_object_type_id_t(src_elem_type_id),
                    st_buffer_size_t(src_buffer_id),
                    st_buffer_size_t(src_elem_index),
                    st_buffer_size_t(src_pointer_offset),
                )
        return has_item

    def index_of_assign_address_item(
        self,
        item=None,
        dest_buffer_id=None,
        src_buffer_id=None,
        dest_elem_type_id=None,
        dest_elem_index=None,
        dest_pointer_offset=None,
        src_elem_type_id=None,
        src_elem_index=None,
        src_pointer_offset=None,
    ):
        index_of_item = None
        if not (item is None):
            if isinstance(item, AssignAddressItem):
                # TODO: Figure out a way to do this without relying on
                #      internal API for cobjects
                index_of_item = st.st_TrackJob_index_of_assign_address_item(
                    self.ptr_st_track_job,
                    ct.cast(item._get_address(), st.st_AssignAddressItem_p),
                )
            elif item != st.st_NullAssignAddressItem:
                index_of_item = st.st_TrackJob_index_of_assign_address_item(
                    self.ptr_st_track_job, item
                )
        elif (
            not (dest_buffer_id is None)
            and not (src_buffer_id is None)
            and not (dest_elem_type_id is None)
            and not (dest_elem_index is None)
            and not (dest_pointer_offset is None)
            and not (src_elem_type_id is None)
            and not (src_elem_index is None)
            and not (src_pointer_offset is None)
        ):
            index_of_item = st.st_TrackJob_index_of_assign_address_item_detailed(
                self.ptr_st_track_job,
                st_object_type_id_t(dest_elem_type_id),
                st_buffer_size_t(dest_buffer_id),
                st_buffer_size_t(dest_elem_index),
                st_buffer_size_t(dest_pointer_offset),
                st_object_type_id_t(src_elem_type_id),
                st_buffer_size_t(src_buffer_id),
                st_buffer_size_t(src_elem_index),
                st_buffer_size_t(src_pointer_offset),
            )
        return index_of_item

    def add_assign_address_item(
        self,
        item=None,
        dest_elem_type_id=None,
        dest_buffer_id=None,
        dest_elem_index=None,
        dest_pointer_offset=None,
        src_elem_type_id=None,
        src_buffer_id=None,
        src_elem_index=None,
        src_pointer_offset=None,
    ):
        ptr_added_item = st.st_NullAssignAddressItem
        if not (item is None):
            if isinstance(item, AssignAddressItem):
                # TODO: Figure out a way to do this without relying on
                #      internal API for cobjects
                _ptr_item = ct.cast(
                    item._get_address(), st.st_AssignAddressItem_p
                )

                ptr_added_item = st.st_TrackJob_add_assign_address_item(
                    self.ptr_st_track_job, _ptr_item
                )
            elif item != st.st_NullAssignAddressItem:
                ptr_added_item = st.st_TrackJob_add_assign_address_item(
                    self.ptr_st_track_job, item
                )
        elif (
            not (dest_elem_type_id is None)
            and not (dest_buffer_id is None)
            and not (dest_elem_index is None)
            and not (dest_pointer_offset is None)
            and not (src_elem_type_id is None)
            and not (src_buffer_id is None)
            and not (src_elem_index is None)
            and not (src_pointer_offset is None)
        ):
            ptr_added_item = st.st_TrackJob_add_assign_address_item_detailed(
                self.ptr_st_track_job,
                st.st_object_type_id_t(dest_elem_type_id),
                st_buffer_size_t(dest_buffer_id),
                st_buffer_size_t(dest_elem_index),
                st_buffer_size_t(dest_pointer_offset),
                st.st_object_type_id_t(src_elem_type_id),
                st_buffer_size_t(src_buffer_id),
                st_buffer_size_t(src_elem_index),
                st_buffer_size_t(src_pointer_offset),
            )

        if ptr_added_item == st.st_NullAssignAddressItem:
            raise ValueError(
                "unable to add AssignAddressItem given by these parameters"
            )
        return ptr_added_item

    def commit_address_assignments(self):
        self._last_status = st.st_TrackJob_commit_address_assignments(
            self.ptr_st_track_job
        )

        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("Unable to commit address assignment items")

        return self

    def assign_all_addresses(self):
        self._last_status = st.st_TrackJob_assign_all_addresses(
            self.ptr_st_track_job
        )

        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError(
                "Unable to perform assignment of all address items"
            )

        return self

    def assign_addresses(self, dest_buffer_id, src_buffer_id):
        if dest_buffer_id is None and src_buffer_id is None:
            self._last_status = st.st_TrackJob_assign_all_addresses(
                self.ptr_st_track_job
            )
        elif not (dest_buffer_id is None) and not (src_buffer_id is None):
            self._last_status = st.st_TrackJob_assign_addresses(
                self.ptr_st_track_job,
                st_buffer_size_t(dest_buffer_id),
                st_buffer_size_t(src_buffer_id),
            )
        else:
            raise ValueError(
                "inconsistent dest_buffer_id and src_buffer_id parameters"
            )
            self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value

        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("Unable to perform assignment of address items")
        return self

    # --------------------------------------------------------------------------

    def reset(
        self,
        beam_elements_buffer,
        particles_buffer,
        until_turn_elem_by_elem=0,
        output_buffer=None,
    ):
        _new_ptr_c_particles_buffer = st.st_NullBuffer
        _new_ptr_c_beam_elements_buffer = st.st_NullBuffer
        _new_ptr_c_output_buffer = st.st_NullBuffer

        if particles_buffer is not None:
            particles_buffer = get_cbuffer_from_obj(particles_buffer)
            _new_ptr_c_particles_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                particles_buffer
            )
            if _new_ptr_c_particles_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input particles buffer")

        if beam_elements_buffer is not None:
            beam_elements_buffer = get_cbuffer_from_obj(beam_elements_buffer)
            _new_ptr_c_beam_elements_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                beam_elements_buffer
            )
            if _new_ptr_c_beam_elements_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input beam elements buffer")

        particles = st.st_Particles_buffer_get_particles(
            _new_ptr_c_particles_buffer, 0
        )

        if particles == st.st_NullParticles:
            raise ValueError("Required particle sets not available")

        until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)

        out_buffer_flags = st.st_OutputBuffer_required_for_tracking(
            particles, _new_ptr_c_beam_elements_buffer, until_turn_elem_by_elem
        )

        needs_output_buffer = st.st_OutputBuffer_requires_output_buffer(
            ct.c_int32(out_buffer_flags)
        )

        if needs_output_buffer:
            num_objects = ct.c_uint64(0)
            num_slots = ct.c_uint64(0)
            num_dataptrs = ct.c_uint64(0)
            num_garbage = ct.c_uint64(0)
            slot_size = st.st_Buffer_get_slot_size(_new_ptr_c_particles_buffer)

            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                _new_ptr_c_beam_elements_buffer,
                particles,
                until_turn_elem_by_elem,
                ct.byref(num_objects),
                ct.byref(num_slots),
                ct.byref(num_dataptrs),
                ct.byref(num_garbage),
                slot_size,
            )

            if ret == 0:
                if (
                    num_objects.value > 0
                    and num_slots.value > 0
                    and num_dataptrs.value > 0
                    and num_garbage.value >= 0
                ):
                    if output_buffer is None:
                        output_buffer = CBuffer(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value,
                        )
                    else:
                        output_buffer.reallocate(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value,
                        )

                    if output_buffer is None:
                        raise ValueError("Could not provide output buffer")

                _new_ptr_c_output_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                    output_buffer
                )
                if _new_ptr_c_output_buffer == st.st_NullBuffer:
                    raise ValueError("Unable to map (optional) output buffer")
            else:
                raise ValueError("Error pre-calculating out buffer params")
        elif output_buffer is not None:
            _new_ptr_c_output_buffer = st.st_Buffer_new_mapped_on_cbuffer(
                output_buffer
            )
            if _new_ptr_c_output_buffer == st.st_NullBuffer:
                raise ValueError("Unable to map (optional) output buffer")

        if (
            self.ptr_st_track_job != st.st_NullTrackJob
            and _new_ptr_c_particles_buffer != st_NullBuffer
            and _new_ptr_c_beam_elements_buffer != st_NullBuffer
        ):

            if st.st_TrackJob_reset_with_output(
                self.ptr_st_track_job,
                _new_ptr_c_particles_buffer,
                _new_ptr_c_beam_elements_buffer,
                _new_ptr_c_output_buffer,
                until_turn_elem_by_elem,
            ):

                if self._ptr_c_particles_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete(self._ptr_c_particles_buffer)
                    self._particles_buffer = particles_buffer
                    self._ptr_c_particles_buffer = _new_ptr_c_particles_buffer

                if self._ptr_c_beam_elements_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete(self._ptr_c_beam_elements_buffer)
                    self._beam_elements_buffer = beam_elements_buffer
                    self._ptr_c_beam_elements_buffer = (
                        _new_ptr_c_beam_elements_buffer
                    )

                if self._output_buffer is not None:
                    del self._output_buffer
                    self._output_buffer = None
                    self._ptr_c_output_buffer = st.st_NullBuffer

                if output_buffer is not None:
                    self._output_buffer = output_buffer
                    self._ptr_c_output_buffer = _new_ptr_c_output_buffer

            else:
                if _new_ptr_c_particles_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete(_new_ptr_c_particles_buffer)

                if _new_ptr_c_beam_elements_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete(_new_ptr_c_beam_elements_buffer)

        return self

    # -------------------------------------------------------------------------

    def add_stored_buffer(
        self,
        cbuffer=None,
        size=None,
        ptr_c_buffer=None,
        take_ownership=False,
        delete_ptr_after_move=False,
    ):
        _st_buffer = None
        buffer_id = st.st_ARCH_ILLEGAL_BUFFER_ID.value
        if not (cbuffer is None):
            _st_buffer = Buffer.from_cbuffer(cbuffer)
        elif not (size is None) and size > 0:
            _st_buffer = Buffer(size=size)
        elif not (ptr_c_buffer is None) and ptr_c_buffer != st_NullBuffer:
            owns_buffer = take_ownership
            owns_pointer = take_ownership and delete_ptr_after_move
            _st_buffer = Buffer(
                ptr_ext_buffer=ptr_c_buffer,
                owns_buffer=owns_buffer,
                owns_pointer=owns_pointer,
            )
        if not (_st_buffer is None) and _st_buffer.pointer != st_NullBuffer:
            buffer_id = st.st_TrackJob_add_stored_buffer(
                self.ptr_st_track_job,
                _st_buffer.pointer,
                ct.c_bool(False),
                ct.c_bool(False),
            )
            if buffer_id != st.st_ARCH_ILLEGAL_BUFFER_ID.value:
                assert buffer_id not in self._stored_buffers
                self._stored_buffers[buffer_id] = _st_buffer
        return buffer_id

    def remove_stored_buffer(self, buffer_id):
        return st.st_TrackJob_remove_stored_buffer(
            self.ptr_st_track_job, st_arch_size_t(buffer_id)
        )

    def owns_stored_buffer(self, buffer_id):
        return st.st_TrackJob_owns_stored_buffer(
            self.ptr_st_track_job, st_arch_size_t(buffer_id)
        )

    @property
    def stored_buffer_is_cbuffer(self, buffer_id):
        return (
            buffer_id in self._stored_buffers
            and self._stored_buffers[buffer_id].maps_to_cbuffer
        )

    def stored_buffer_cbuffer(self, buffer_id):
        if not self.stored_buffer_is_cbuffer:
            raise RuntimeError(
                f"Unable to retrieve CBuffer for buffer_id={buffer_id}"
            )
        return self._stored_buffers[buffer_id].cbuffer

    def ptr_stored_buffer(self, buffer_id):
        return st.st_TrackJob_stored_buffer(
            self.ptr_st_track_job, st_arch_size_t(buffer_id)
        )

    def stored_buffer(self, buffer_id):
        return self._stored_buffers.get(buffer_id, None)

    def push_stored_buffer(self, buffer_id):
        self._last_status = st.st_TrackJob_push_stored_buffer(
            self.ptr_st_track_job, st_buffer_size_t(buffer_id)
        )
        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError(f"Unable to push stored buffer {buffer_id}")
        return self

    def collect_stored_buffer(self, buffer_id):
        self._last_status = st.st_TrackJob_collect_stored_buffer(
            self.ptr_st_track_job, st_buffer_size_t(buffer_id)
        )
        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError(f"Unable to collect stored buffer {buffer_id}")
        return self

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

    def argument_by_buffer_id(self, buffer_id):
        ptr_arg = st.st_NullClArgument.value
        if self.arch_str == "opencl":
            ptr_arg = st.st_TrackJobCl_argument_by_buffer_id(
                self.ptr_st_track_job, st_buffer_size_t(buffer_id)
            )
        else:
            arch_str = self.arch_str
            raise RuntimeError(
                f"unable to get argument for buffer on arch {arch_str}"
            )
        return ptr_arg

    def stored_buffer_argument(self, buffer_id):
        ptr_arg = st.st_NullClArgument.value
        if self.arch_str == "opencl":
            ptr_arg = st.st_TrackJobCl_stored_buffer_argument(
                self.ptr_st_track_job, st_buffer_size_t(buffer_id)
            )
        else:
            arch_str = self.arch_str
            raise RuntimeError(
                f"unable to get argument for stored buffer on arch {arch_str}"
            )
        return ptr_arg


# end: python/sixtracklib/trackjob.py
