#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer, CObject

from . import stcommon as st
from . import config as stconf
from .particles import ParticlesSet


def _get_buffer(obj):
    if isinstance(obj, CBuffer):
        return obj
    elif isinstance(obj, CObject):
        return obj._buffer
    elif hasattr(obj, 'cbuffer'):
        return obj.cbuffer
    else:
        raise ValueError("Object {obj} is not or has not a CBuffer")


class TrackJob(object):
    @staticmethod
    def enabled_archs():
        enabled_archs = [arch_str for arch_str, flag in
                         stconf.SIXTRACKLIB_MODULES.items() if flag]
        if 'cpu' not in stconf.SIXTRACKLIB_MODULES:
            enabled_archs.append("cpu")
        return enabled_archs

    @staticmethod
    def print_nodes(arch_str):
        arch_str = arch_str.strip().lower()
        if stconf.SIXTRACKLIB_MODULES.get(arch_str, False):
            if arch_str == "opencl":
                context = st.st_ClContext_create()
                st.st_ClContextBase_print_nodes_info(context)
                st.st_ClContextBase_delete(context)
            else:
                print("nodes not available for architecture {0}".format(
                    arch_str))
        else:
            print("architecture {0} is not enabled/known".format(arch_str))

    def __init__(self,
                 beam_elements_buffer,
                 particles_buffer,
                 until_turn_elem_by_elem=0,
                 arch='cpu',
                 device_id=None,
                 device=None,
                 output_buffer=None,
                 config_str=None):
        self.ptr_st_track_job = st.st_NullTrackJob
        self._particles_buffer = None
        self._ptr_c_particles_buffer = st.st_NullBuffer
        self._beam_elements_buffer = None
        self._ptr_c_beam_elements_buffer = st.st_NullBuffer
        self._output_buffer = None
        self._ptr_c_output_buffer = st.st_NullBuffer

        base_addr_t = ct.POINTER(ct.c_ubyte)
        success = False

        if particles_buffer is not None:
            particles_buffer = _get_buffer(particles_buffer)
            self._particles_buffer = particles_buffer
            self._ptr_c_particles_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(particles_buffer)
            if self._ptr_c_particles_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input particles buffer")

        if beam_elements_buffer is not None:
            beam_elements_buffer = _get_buffer(beam_elements_buffer)
            self._beam_elements_buffer = beam_elements_buffer
            self._ptr_c_beam_elements_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(beam_elements_buffer)
            if self._ptr_c_beam_elements_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input beam elements buffer")

        particles = st.st_Particles_buffer_get_particles(
            self._ptr_c_particles_buffer, 0)

        if particles == st.st_NullParticles:
            raise ValueError("Required particle sets not available")

        until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)
        out_buffer_flags = st.st_OutputBuffer_required_for_tracking(
            particles, self._ptr_c_beam_elements_buffer, until_turn_elem_by_elem)
        needs_output_buffer = st.st_OutputBuffer_requires_output_buffer(
            ct.c_int32(out_buffer_flags))

        if needs_output_buffer:
            num_objects = ct.c_uint64(0)
            num_slots = ct.c_uint64(0)
            num_dataptrs = ct.c_uint64(0)
            num_garbage = ct.c_uint64(0)
            slot_size = st.st_Buffer_get_slot_size(
                self._ptr_c_particles_buffer)

            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                self._ptr_c_beam_elements_buffer, particles,
                until_turn_elem_by_elem, ct.byref(num_objects),
                ct.byref(num_slots), ct.byref(num_dataptrs),
                ct.byref(num_garbage), slot_size)

            if ret == 0:
                if num_objects.value > 0 and num_slots.value > 0 and \
                        num_dataptrs.value > 0 and num_garbage.value >= 0:
                    if output_buffer is None:
                        output_buffer = CBuffer(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value)
                    else:
                        output_buffer.reallocate(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value)

                    if output_buffer is None:
                        raise ValueError("Could not provide output buffer")

                self._output_buffer = output_buffer
                self._ptr_c_output_buffer = \
                    st.st_Buffer_new_mapped_on_cbuffer(output_buffer)
                if self._ptr_c_output_buffer == st.st_NullBuffer:
                    raise ValueError("Unable to map (optional) output buffer")
            else:
                raise ValueError("Error pre-calculating out buffer params")
        elif output_buffer is not None:
            self._output_buffer = output_buffer
            self._ptr_c_output_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(self._output_buffer)
            if self._ptr_c_output_buffer == st.st_NullBuffer:
                raise ValueError("Unable to map (optional) output buffer")

        assert((needs_output_buffer and
                self._ptr_c_output_buffer != st.st_NullBuffer) or
               (not needs_output_buffer))

        if device is not None:
            arch, device_id = device.split(':')

        arch = arch.strip().lower()
        if not(stconf.SIXTRACKLIB_MODULES.get(arch, False) is not False
                or arch == 'cpu'):
            raise ValueError("Unknown architecture {0}".format(arch, ))

        if device_id is not None:
            if config_str is None:
                config_str = device_id
            else:
                config_str = device_id + ";" + config_str
        else:
            config_str = ""

        arch = arch.encode('utf-8')
        config_str = config_str.encode('utf-8')

        self.ptr_st_track_job = st.st_TrackJob_new_with_output(
            ct.c_char_p(arch), self._ptr_c_particles_buffer,
            self._ptr_c_beam_elements_buffer, self._ptr_c_output_buffer,
            until_turn_elem_by_elem, ct.c_char_p(config_str))

        if self.ptr_st_track_job == st.st_NullTrackJob:
            raise ValueError('unable to construct TrackJob from arguments')

    def __del__(self):
        if self.ptr_st_track_job != st.st_NullTrackJob:
            job_owns_output_buffer = st.st_TrackJob_owns_output_buffer(
                self.ptr_st_track_job)

            st.st_TrackJob_delete(self.ptr_st_track_job)
            self.ptr_st_track_job = st.st_NullTrackJob

            if job_owns_output_buffer and \
                    self._ptr_c_output_buffer != st.st_NullBuffer:
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
        return st.st_TrackJob_track_until(
            self.ptr_st_track_job, ct.c_uint64(until_turn))

    def track_elem_by_elem(self, until_turn):
        return st.st_TrackJob_track_elem_by_elem(
            self.ptr_st_track_job, ct.c_uint64(until_turn))

    def track_line(self, begin_idx, end_idx, finish_turn=False):
        return st.st_TrackJob_track_line(
            self.ptr_st_track_job,
            ct.c_uint64(begin_idx),
            ct.c_uint64(end_idx),
            ct.c_bool(finish_turn))

    def collect(self):
        st.st_TrackJob_collect(self.ptr_st_track_job)
        return

    def type(self):
        return st.st_TrackJob_get_type_id(self.ptr_st_track_job)

    def type_str(self):
        str = st.st_TrackJob_get_type_str(self.ptr_st_track_job)
        return str.decode('utf-8')

    def num_beam_monitors(self):
        return st.st_TrackJob_get_num_beam_monitors(self.ptr_st_track_job)

    def has_elem_by_elem_output(self):
        return st.st_TrackJob_has_elem_by_elem_output(self.ptr_st_track_job)

    def has_beam_monitor_output(self):
        return st.st_TrackJob_has_beam_monitor_output(self.ptr_st_track_job)

    def elem_by_elem_output_offset(self):
        return st.st_TrackJob_get_elem_by_elem_output_buffer_offset(
            self.ptr_st_track_job)

    def beam_monitor_output_offset(self):
        return st.st_TrackJob_get_beam_monitor_output_buffer_offset(
            self.ptr_st_track_job)

    def has_output_buffer(self):
        return st.st_TrackJob_has_output_buffer(self.ptr_st_track_job) and \
            bool(self._output_buffer is not None)

    def reset(self, beam_elements_buffer, particles_buffer,
                  until_turn_elem_by_elem=0, output_buffer=None):
        _new_ptr_c_particles_buffer = st.st_NullBuffer
        _new_ptr_c_beam_elements_buffer = st.st_NullBuffer
        _new_ptr_c_output_buffer = st.st_NullBuffer

        if particles_buffer is not None:
            particles_buffer = _get_buffer(particles_buffer)
            _new_ptr_c_particles_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(particles_buffer)
            if _new_ptr_c_particles_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input particles buffer")

        if beam_elements_buffer is not None:
            beam_elements_buffer = _get_buffer(beam_elements_buffer)
            _new_ptr_c_beam_elements_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(beam_elements_buffer)
            if _new_ptr_c_beam_elements_buffer == st.st_NullBuffer:
                raise ValueError("Issues with input beam elements buffer")

        particles = st.st_Particles_buffer_get_particles(
            _new_ptr_c_particles_buffer, 0)

        if particles == st.st_NullParticles:
            raise ValueError("Required particle sets not available")

        until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)

        out_buffer_flags = st.st_OutputBuffer_required_for_tracking(
            particles, _new_ptr_c_beam_elements_buffer, until_turn_elem_by_elem)

        needs_output_buffer = st.st_OutputBuffer_requires_output_buffer(
            ct.c_int32(out_buffer_flags))

        if needs_output_buffer:
            num_objects = ct.c_uint64(0)
            num_slots = ct.c_uint64(0)
            num_dataptrs = ct.c_uint64(0)
            num_garbage = ct.c_uint64(0)
            slot_size = st.st_Buffer_get_slot_size(_new_ptr_c_particles_buffer)

            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                _new_ptr_c_beam_elements_buffer, particles,
                until_turn_elem_by_elem, ct.byref(num_objects),
                ct.byref(num_slots), ct.byref(num_dataptrs),
                ct.byref(num_garbage), slot_size)

            if ret == 0:
                if num_objects.value > 0 and num_slots.value > 0 and \
                        num_dataptrs.value > 0 and num_garbage.value >= 0:
                    if output_buffer is None:
                        output_buffer = CBuffer(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value)
                    else:
                        output_buffer.reallocate(
                            max_slots=num_slots.value,
                            max_objects=num_objects.value,
                            max_pointers=num_dataptrs.value,
                            max_garbage=num_garbage.value)

                    if output_buffer is None:
                        raise ValueError("Could not provide output buffer")

                _new_ptr_c_output_buffer = \
                    st.st_Buffer_new_mapped_on_cbuffer(output_buffer)
                if _new_ptr_c_output_buffer == st.st_NullBuffer:
                    raise ValueError("Unable to map (optional) output buffer")
            else:
                raise ValueError("Error pre-calculating out buffer params")
        elif output_buffer is not None:
            _new_ptr_c_output_buffer = \
                st.st_Buffer_new_mapped_on_cbuffer(output_buffer)
            if _new_ptr_c_output_buffer == st.st_NullBuffer:
                raise ValueError("Unable to map (optional) output buffer")

        if self.ptr_st_track_job != st.st_NullTrackJob and \
            _new_ptr_c_particles_buffer != st_NullBuffer and \
            _new_ptr_c_beam_elements_buffer != st_NullBuffer:

            if st.st_TrackJob_reset_with_output(self.ptr_st_track_job,
                _new_ptr_c_particles_buffer, _new_ptr_c_beam_elements_buffer,
                _new_ptr_c_output_buffer, until_turn_elem_by_elem):

                if self._ptr_c_particles_buffer != st.st_NullBuffer:
                   st.st_Buffer_delete( self._ptr_c_particles_buffer )
                   self._particles_buffer = particles_buffer
                   self._ptr_c_particles_buffer = _new_ptr_c_particles_buffer

                if self._ptr_c_beam_elements_buffer != st.st_NullBuffer:
                   st.st_Buffer_delete( self._ptr_c_beam_elements_buffer )
                   self._beam_elements_buffer = beam_elements_buffer
                   self._ptr_c_beam_elements_buffer = _new_ptr_c_beam_elements_buffer

                if self._output_buffer is not None:
                    del self._output_buffer
                    self._output_buffer = None
                    self._ptr_c_output_buffer = st.st_NullBuffer

                if output_buffer is not None:
                    self._output_buffer = output_buffer
                    self._ptr_c_output_buffer = _new_ptr_c_output_buffer

            else:
                if _new_ptr_c_particles_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete( _new_ptr_c_particles_buffer )

                if _new_ptr_c_beam_elements_buffer != st.st_NullBuffer:
                    st.st_Buffer_delete( _new_ptr_c_beam_elements_buffer )

        return self

# end: python/pysixtracklib/trackjob.py
