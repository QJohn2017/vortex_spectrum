from core import BeamXY, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, VisualizerXY, parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamXY(medium='LiF',
              p_0_to_p_vortex=5,
              m=1,
              M=1,
              lmbda=1800*10**-9,
              x_0=100*10**-6,
              y_0=300*10**-6,
              radii_in_grid=70,  # 70 # 140 # 170 #8
              noise_percent=0.0,
              n_x=4096,  # 8k
              n_y=4096)

# create visualizer object
visualizer = VisualizerXY(beam=beam,
                          remaining_central_part_coeff_field=0.05, #0.3,  # 0.04# 0.07 # 0.03
                          remaining_central_part_coeff_spectrum=0.05) #0.03)  # 0.08 # 0.015 # 0.03

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                        kerr_effect=KerrExecutorXY(beam=beam),
                        n_z=0,
                        dz_0=beam.z_diff / 1000,
                        const_dz=True,
                        print_current_state_every=1,
                        plot_beam_every=1,
                        max_intensity_to_stop=10**17,
                        visualizer=visualizer)

# initiate propagation process
propagator.propagate()
