#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sixtracklib as pyst

if __name__ == '__main__':

    # ========================================================================
    # Setup command line options and arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("architecture", help="Architecture to be used")

    parser.add_argument("-d", "--device", help="Device id string", default="",
                        dest="device_id", required=False, )

    parser.add_argument("-p", "--particles",
                        help="Path to particles dump file",
                        dest="particles_buffer", required=True, )

    parser.add_argument("-e", "--beam-elements",
                        help="Path to beam-elements dump file",
                        dest="beam_elements_buffer", required=True, )

    parser.add_argument(
        "-E",
        "--until_turn_elem_by_elem",
        help="Dump all particles element by element wise until this turn",
        required=False,
        default=0,
        type=int,
        dest="until_turn_elem_by_elem")

    parser.add_argument(
        "-T",
        "--until_turn_turn_by_turn",
        help="Dump the state of all particles at the end of each turn" +
        ";Only applies if not alreay performing a diffferent output operation",
        required=False,
        default=0,
        type=int,
        dest="until_turn_turn_by_turn")

    parser.add_argument(
        "-t",
        "--until_turn_output",
        help="Dump the state of all particles at the end of each turn " +
        "every --skip_out_turns turn",
        required=False,
        default=0,
        type=int,
        dest="until_turn_output")

    parser.add_argument(
        "-s",
        "--skip_out_turns",
        help="Number of turns between two consequetive outputs for " +
        "with --until_turn_output; has no effect on other output modes",
        type=int,
        required=False,
        default=1,
        dest="skip_out_turns")

    args = parser.parse_args()

    # ========================================================================
    # Setup input buffers and track job

    path_to_pb = args.particles_buffer
    particles_set = pyst.ParticlesSet.fromfile(path_to_pb)

    path_to_eb = args.beam_elements_buffer
    beam_elements = pyst.Elements.fromfile(path_to_eb)

    num_beam_monitors = pyst.append_beam_monitors_to_lattice(
        beam_elements.cbuffer,
        args.until_turn_elem_by_elem,
        args.until_turn_turn_by_turn,
        args.until_turn_output,
        args.skip_out_turns)

    print("Added {0} beam monitors to the lattice".format(num_beam_monitors))

    # =======================================================================
    # Create the TrackJob instance

    job = pyst.TrackJob(beam_elements.cbuffer, particles_set.cbuffer,
                        until_turn_elem_by_elem=args.until_turn_elem_by_elem,
                        arch=args.architecture, device_id=args.device_id)

    # ========================================================================
    # Print summary about configuration

    print("TrackJob:")
    print("----------------------------------------------------------------")
    print("architecture                 : {0}".format(job.arch_str))

    if job.has_elem_by_elem_output:
        print("has elem_by_elem output      : yes")
    else:
        print("has elem_by_elem output      : no")

    if job.has_beam_monitor_output:
        print("has beam monitor output      : yes")
        print("num beam monitors            : {0}".format(
            job.num_beam_monitors))
    else:
        print("has beam monitor output      : no")

    if job.has_output_buffer:
        print("has output buffer            : yes")
    else:
        print("has output buffer            : no")

    print(
        "dump elem-by-elem until turn : {0}".format(args.until_turn_elem_by_elem))
    print(
        "dump turn-by-turn until turn : {0}".format(args.until_turn_turn_by_turn))
    print("traget num output turns      : {0}".format(args.until_turn_output))
    print("dump every number of turns   : {0}".format(args.skip_out_turns))

    # ========================================================================
    # Perform tracking

    success = True
    if args.until_turn_elem_by_elem > 0:
        print("\r\ntracking until turn {0} element by element ... ".format(
            args.until_turn_elem_by_elem))

        status = job.track_elem_by_elem(args.until_turn_elem_by_elem)
        success = bool(status == 0)

        print("{0}".format(success and "SUCCESS" or "FAILURE"))

    if success and args.until_turn_output > args.until_turn_elem_by_elem:
        print("\r\ntracking until turn {0} ... ".format(
            args.until_turn_output))

        status = job.track_until(args.until_turn_output)
        print(status)
        success = bool(status == 0)

        print("{0}".format(success and "SUCCESS" or "FAILURE"))

    # ========================================================================
    # Collect data before accessing particle_buffer and output_buffer

    if success:
        job.collect()

    # ========================================================================
    # Access the output buffer

    output_buffer = None

    if success and job.has_output_buffer:
        output_buffer = job.output_buffer

        if job.has_elem_by_elem_output:
            assert(output_buffer.n_objects > job.elem_by_elem_output_offset)
            # These are the particles containing the elem by elem information
            elem_by_elem_particles = output_buffer.get_object(
                job.elem_by_elem_output_offset, cls=pyst.Particles)

        if job.has_beam_monitor_output:
            out_offset = job.beam_monitor_output_offset
            num_monitors = job.num_beam_monitors
            for ii in range(out_offset, num_monitors + out_offset):
                assert(ii < output_buffer.n_objects)
                out_particles = output_buffer.get_object(
                    ii, cls=pyst.Particles)

    # finished
