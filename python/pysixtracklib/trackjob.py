#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer
import pdb

from . import stcommon as st
from . import sixtracklibconf as stconf

class TrackJob(object):
    @staticmethod
    def enabled_archs():
        enabled_archs = [ arch_str for arch_str, flag in
                stconf.SIXTRACKLIB_MODULES.items() if flag ]
        if  not stconf.SIXTRACKLIB_MODULES.get( 'cpu', False ):
            enabled_archs.append( "cpu" )
        return enabled_archs

    @staticmethod
    def print_nodes( arch_str ):
        arch_str = arch_str.strip().lower()
        if  stconf.SIXTRACKLIB_MODULES.get( arch_str, False ):
            if  arch_str == "opencl":
                context = st.st_ClContext_create()
                st.st_ClContextBase_print_nodes_info( context )
                st.st_ClContextBase_delete( context )
            else:
                print( "nodes not available for architecture {0}".format(
                        arch_str ) )
        else:
            print( "architecture {0} is not enabled/known".format( arch_str ) )

    def __init__( self, arch_str, device_id_str=None, particles_buffer=None,
                  beam_elements_buffer=None, output_buffer=None,
                  dump_elem_by_elem_turns=0, config_str=None ):
        self.ptr_st_track_job            = st.st_Null
        self._particles_buffer           = None
        self._ptr_c_particles_buffer     = st.st_Null
        self._beam_elements_buffer       = None
        self._ptr_c_beam_elements_buffer = st.st_Null
        self._output_buffer              = None
        self._ptr_c_output_buffer        = st.st_Null

        base_addr_t = ct.POINTER( ct.c_ubyte )
        success = False

        if particles_buffer is not None:
            self._particles_buffer   = particles_buffer
            ptr = ct.cast( particles_buffer.base, base_addr_t )
            nn  = ct.c_uint64( particles_buffer.size )
            self._ptr_c_particles_buffer = st.st_Buffer_new_on_data( ptr, nn )
            success = bool( self._ptr_c_particles_buffer != st.st_Null )

        if success and beam_elements_buffer is not None:
            self._beam_elements_buffer   = beam_elements_buffer
            ptr = ct.cast( beam_elements_buffer.base, base_addr_t )
            nn  = ct.c_uint64( beam_elements_buffer.size )
            self._ptr_c_beam_elements_buffer = st.st_Buffer_new_on_data( ptr, nn )
            success = bool( self._ptr_c_beam_elements_buffer != st.st_Null )

        if success and self._ptr_c_particles_buffer != st.st_Null and \
            self._ptr_c_beam_elements_buffer != st.st_Null:

            num_particle_sets = ct.c_uint64( 1 )
            pset_index   = ct.c_uint64( 0 )

            num_objects  = ct.c_uint64( 0 )
            num_slots    = ct.c_uint64( 0 )
            num_dataptrs = ct.c_uint64( 0 )
            num_garbage  = ct.c_uint64( 0 )
            num_elem_by_elem_turns = ct.c_uint64( dump_elem_by_elem_turns )

            slot_size    = st.st_Buffer_get_slot_size( self._ptr_c_particles_buffer )
            pdb.set_trace()
            ret = st.st_OutputBuffer_calculate_output_buffer_params(
                self._ptr_c_beam_elements_buffer, self._ptr_c_particles_buffer,
                ct.c_uint64( 1 ), ct.byref( pset_index ),
                num_elem_by_elem_turns, ct.byref( num_objects ),
                ct.byref( num_slots ), ct.byref( num_dataptrs ),
                ct.byref( num_garbage ), ct.c_uint64( slot_size ) )

            if ret == 0 and num_objects.value > 0 and num_slots.value > 0 and \
                num_dataptrs.value > 0 and num_garbage.value >= 0:
                if  output_buffer is None:
                    output_buffer = CBuffer( max_slots=num_slots.value,
                        max_objects=num_objects.value,
                        max_pointers=num_dataptrs.value,
                        max_garbage=num_garbage.value )
                else:
                    output_buffer.allocate( max_slots=num_slots.value,
                        max_objects=num_objects.value,
                        max_pointers=num_dataptrs.value,
                        max_garbage=num_garbage.value )

                assert( output_buffer is not None )
                self._output_buffer = output_buffer
                ptr = ct.cast( output_buffer.base, base_addr_t )
                nn  = ct.c_uint64( output_buffer.size )
                self._ptr_c_output_buffer = st.st_Buffer_new_on_data( ptr, nn )
                success = bool( self._ptr_c_output_buffer != st.st_Null )
            elif ret != 0:
                success = False

        if  success:
            arch_str = arch_str.strip().lower()
            success = bool( stconf.SIXTRACKLIB_MODULES.get(
                    arch_str, False ) or arch_str == 'cpu' )

        if  success:
            if device_id_str is not None:
                if config_str is None:
                    config_str = device_id_str
                else:
                    config_str = device_id_str + ";" + config_str
            else:
                config_str = ""

            arch_str   = arch_str.encode( 'utf-8' )
            config_str = config_str.encode( 'utf-8' )

            self.ptr_st_track_job = st.st_TrackJob_new_with_output(
                ct.c_char_p( arch_str ), self._ptr_c_particles_buffer,
                self._ptr_c_beam_elements_buffer, self._ptr_c_output_buffer,
                num_elem_by_elem_turns, ct.c_char_p( config_str ) )

        if not success or self.ptr_st_track_job == st.st_Null:
            raise ValueError( 'unable to construct TrackJob from arguments' )


    def __del__( self ):
        if  self.ptr_st_track_job != st.st_Null:
            job_owns_output_buffer = st.st_TrackJob_owns_output_buffer(
                self.ptr_st_track_job )

            st.st_TrackJob_delete( self.ptr_st_track_job )
            self.ptr_st_track_job = st.st_Null

            if  job_owns_output_buffer and \
                self._ptr_c_output_buffer != st.st_Null:
                self._ptr_c_output_buffer  = st.st_Null

        if  self._ptr_c_particles_buffer != st.st_Null:
            st.st_Buffer_delete( self._ptr_c_particles_buffer )
            self._ptr_c_particles_buffer = st.st_Null

        if  self._ptr_c_beam_elements_buffer != st.st_Null:
            st.st_Buffer_delete( self._ptr_c_beam_elements_buffer )
            self._ptr_c_beam_elements_buffer  = st.st_Null

        if  self._ptr_c_output_buffer != st.st_Null:
            st.st_Buffer_delete( self._ptr_c_output_buffer )
            self._ptr_c_output_buffer  = st.st_Null


    def track( self, until_turn ):
        return st.st_TrackJob_track_elem_by_elem(
                self.ptr_st_track_job, until_turn )

    def track_elem_by_elem( self, until_turn ):
        return st.st_TrackJob_track_until( self.ptr_st_track_job, until_turn )

    def collect( self ):
        st.st_TrackJob_collect( self.ptr_st_track_job )
        return

    def type( self ):
        return st.st_TrackJob_get_type_id( self.ptr_st_track_job )

    def type_str( self ):
        return st.st_TrackJob_get_type_str( self.ptr_st_track_job )

    def num_beam_monitors( self ):
        return st.st_TrackJob_get_num_beam_monitors( self.ptr_st_track_job )

    def has_elem_by_elem_outupt( self ):
        return st.st_TrackJob_has_elem_by_elem_output( self.ptr_st_track_job )

    def elem_by_elem_output_offset( self ):
        return st.st_TrackJob_get_elem_by_elem_output_buffer_offset(
            self.ptr_st_track_job )

    def beam_monitor_output_offset( self ):
        return st.st_TrackJob_get_beam_monitor_output_buffer_offset(
            self.ptr_st_track_job )

# end: python/pysixtracklib/trackjob.py
