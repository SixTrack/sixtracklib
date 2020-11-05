if(  NOT SETUP_SIXTRL_FEATURES_FINISHED )
    set( SETUP_SIXTRL_FEATURES_FINISHED 1 )

    # -------------------------------------------------------------------------
    # Track features setup

    set_property( CACHE SIXTRACKL_TRACK_BEAMBEAM4D
                        SIXTRACKL_TRACK_BEAMBEAM6D
                        SIXTRACKL_TRACK_SPACECHARGE
                        SIXTRACKL_TRACK_TRICUB
                  PROPERTY STRINGS enabled disabled skip )

    set( SIXTRL_TRACK_MAP_ENABLED_VALUE  "2" )
    set( SIXTRL_TRACK_MAP_SKIP_VALUE     "1" )
    set( SIXTRL_TRACK_MAP_DISABLED_VALUE "0" )

    set_property( CACHE SIXTRACKL_TRACK_BEAMBEAM4D PROPERTY HELPSTRING
                  "Track over beam-beam 4D beam element" )

    if( SIXTRACKL_TRACK_BEAMBEAM4D )
        if( "${SIXTRACKL_TRACK_BEAMBEAM4D}" STREQUAL "skip" )
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG "${SIXTRL_TRACK_MAP_SKIP_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR "skip" )
        elseif( "${SIXTRACKL_TRACK_BEAMBEAM4D}" STREQUAL "disabled" )
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG "${SIXTRL_TRACK_MAP_DISABLED_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR "disabled" )
        else()
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG "${SIXTRL_TRACK_MAP_ENABLED_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR "enabled" )
        endif()
    endif()

    set_property( CACHE SIXTRACKL_TRACK_BEAMBEAM6D PROPERTY HELPSTRING
                  "Track over beam-beam 6D beam element" )

    if( SIXTRACKL_TRACK_BEAMBEAM6D )
        if( "${SIXTRACKL_TRACK_BEAMBEAM6D}" STREQUAL "skip" )
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG "${SIXTRL_TRACK_MAP_SKIP_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG_STR "skip" )
        elseif( "${SIXTRACKL_TRACK_BEAMBEAM6D}" STREQUAL "disabled" )
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG "${SIXTRL_TRACK_MAP_DISABLED_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG_STR "disabled" )
        else()
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG "${SIXTRL_TRACK_MAP_ENABLED_VALUE}" )
            set( SIXTRL_TRACK_BEAMBEAM6D_FLAG_STR "enabled" )
        endif()
    endif()

    set_property( CACHE SIXTRACKL_TRACK_SPACECHARGE PROPERTY HELPSTRING
                  "Track over frozen space-charge beam element" )

    if( SIXTRACKL_TRACK_SPACECHARGE )
        if( "${SIXTRACKL_TRACK_SPACECHARGE}" STREQUAL "skip" )
            set( SIXTRL_TRACK_SC_FLAG "${SIXTRL_TRACK_MAP_SKIP_VALUE}" )
            set( SIXTRL_TRACK_SC_FLAG_STR "skip" )
        elseif( "${SIXTRACKL_TRACK_SPACECHARGE}" STREQUAL "disabled" )
            set( SIXTRL_TRACK_SC_FLAG "${SIXTRL_TRACK_MAP_DISABLED_VALUE}" )
            set( SIXTRL_TRACK_SC_FLAG_STR "disabled" )
        else()
            set( SIXTRL_TRACK_SC_FLAG "${SIXTRL_TRACK_MAP_ENABLED_VALUE}" )
            set( SIXTRL_TRACK_SC_FLAG_STR "enabled" )
        endif()
    endif()

    set_property( CACHE SIXTRACKL_TRACK_TRICUB PROPERTY HELPSTRING
                  "Track over tri-cub interpolation beam elements" )

    if( SIXTRACKL_TRACK_TRICUB )
        if( "${SIXTRACKL_TRACK_TRICUB}" STREQUAL "skip" )
            set( SIXTRL_TRACK_TRICUB_FLAG "${SIXTRL_TRACK_MAP_SKIP_VALUE}" )
            set( SIXTRL_TRACK_TRICUB_FLAG_STR "skip" )
        elseif( "${SIXTRACKL_TRACK_TRICUB}" STREQUAL "disabled" )
            set( SIXTRL_TRACK_TRICUB_FLAG "${SIXTRL_TRACK_MAP_DISABLED_VALUE}" )
            set( SIXTRL_TRACK_TRICUB_FLAG_STR "disabled" )
        else()
            set( SIXTRL_TRACK_TRICUB_FLAG "${SIXTRL_TRACK_MAP_ENABLED_VALUE}" )
            set( SIXTRL_TRACK_TRICUB_FLAG_STR "enabled" )
        endif()
    endif()

    set( SIXTRL_TRACK_FEATURES_INSTALL_STR
         "set( SIXTRL_TRACK_MAP_ENABLED_VALUE    \"${SIXTRL_TRACK_MAP_ENABLED_VALUE}\" )
          set( SIXTRL_TRACK_MAP_SKIP_VALUE       \"${SIXTRL_TRACK_MAP_SKIP_VALUE}\" )
          set( SIXTRL_TRACK_MAP_DISABLED_VALUE   \"${SIXTRL_TRACK_MAP_DISABLED_VALUE}\" )
          set( SIXTRL_TRACK_BEAMBEAM4D_FLAG      \"${SIXTRL_TRACK_BEAMBEAM4D_FLAG}\" )
          set( SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR  \"${SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR}\" )
          set( SIXTRL_TRACK_BEAMBEAM6D_FLAG      \"${SIXTRL_TRACK_BEAMBEAM4D_FLAG}\" )
          set( SIXTRL_TRACK_BEAMBEAM6D_FLAG_STR  \"${SIXTRL_TRACK_BEAMBEAM4D_FLAG_STR}\" )
          set( SIXTRL_TRACK_SPACECHARGE_FLAG     \"${SIXTRL_TRACK_SPACECHARGE_FLAG}\" )
          set( SIXTRL_TRACK_SPACECHARGE_FLAG_STR \"${SIXTRL_TRACK_SPACECHARGE_FLAG_STR}\" )
          set( SIXTRL_TRACK_TRICUB_FLAG          \"${SIXTRL_TRACK_TRICUB_FLAG}\"
          set( SIXTRL_TRACK_TRICUB_FLAG_STR      \"${SIXTRL_TRACK_TRICUB_FLAG_STR}\" )" )


    # -------------------------------------------------------------------------
    # Aperture check features:

    set_property( CACHE SIXTRACKL_APERTURE_CHECK_AT_DRIFT PROPERTY HELPSTRING
                  "Perform an x-y aperture check at Drift and DriftExact beam elements" )

    set_property( CACHE SIXTRACKL_APERTURE_CHECK_AT_DRIFT
                  PROPERTY STRINGS always conditional never )

    set( SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE        "2" )
    set( SIXTRL_GLOBAL_APERATURE_CHECK_CONDITIONAL_VALUE   "1" )
    set( SIXTRL_GLOBAL_APERATURE_CHECK_NEVER_VALUE         "0" )

    if( SIXTRACKL_APERTURE_CHECK_AT_DRIFT )
        if( "${SIXTRACKL_APERTURE_CHECK_AT_DRIFT}" STREQUAL "never" )
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG
                "${SIXTRL_GLOBAL_APERATURE_CHECK_NEVER_VALUE}" )
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG_STR "never" )
        elseif( "${SIXTRACKL_APERTURE_CHECK_AT_DRIFT}" STREQUAL "conditional" )
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG
                "${SIXTRL_GLOBAL_APERATURE_CHECK_CONDITIONAL_VALUE}" )
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG_STR "conditional" )
        else()
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG
                "${SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE}" )
            set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG_STR "always" )
        endif()
    endif()

    set_property( CACHE SIXTRACKL_APERTURE_X_LIMIT_VALUE PROPERTY ADVANCED )
    set_property( CACHE SIXTRACKL_APERTURE_X_LIMIT_VALUE PROPERTY HELPSTRING
                  "Global aperture check limit for x [m]" )

    set_property( CACHE SIXTRACKL_APERTURE_Y_LIMIT_VALUE PROPERTY ADVANCED )
    set_property( CACHE SIXTRACKL_APERTURE_Y_LIMIT_VALUE PROPERTY HELPSTRING
                  "Global aperture check limit for y [m]" )

    set_property( CACHE SIXTRACKL_APERTURE_CHECK_MIN_DRIFT_LENGTH
                  PROPERTY HELPSTRING
                  "Perform conditional aperture checks for Drift and DriftExact elements with lengths larger than this [m]" )

    set_property( CACHE SIXTRACKL_APERTURE_CHECK_MIN_DRIFT_LENGTH
                  PROPERTY ADVANCED )

    set( SIXTRL_APERTURE_CHECK_FEATURES_INSTALL_STR
         "set( SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE
           \"${SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE}\" )
          set( SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE_STR
           \"${SIXTRL_GLOBAL_APERATURE_CHECK_ALWAYS_VALUE_STR}\" )
          set( SIXTRL_GLOBAL_APERATURE_CHECK_CONDITIONAL_VALUE
           \"${SIXTRL_GLOBAL_APERATURE_CHECK_CONDITIONAL_VALUE}\" )
          set( SIXTRL_GLOBAL_APERATURE_CHECK_NEVER_VALUE
           \"${SIXTRL_GLOBAL_APERATURE_CHECK_NEVER_VALUE}\" )
          set( SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG
           \"${SIXTRL_APERTURE_CHECK_AT_DRIFT_FLAG}\" )
          set( SIXTRACKL_APERTURE_X_LIMIT_VALUE
           \"${SIXTRACKL_APERTURE_X_LIMIT_VALUE}\" )
          set( SIXTRACKL_APERTURE_Y_LIMIT_VALUE
           \"${SIXTRACKL_APERTURE_Y_LIMIT_VALUE}\" )
          set( SIXTRACKL_APERTURE_CHECK_MIN_DRIFT_LENGTH
           \"${SIXTRACKL_APERTURE_CHECK_MIN_DRIFT_LENGTH}\" )" )

endif()
