# CMakeLists.txt for 3D-RT (HEALPix directory)
# --------------------------------------------



# Create a HEALPix library

add_library( _HEALPix   ang2pix_nest.cpp pix2vec_nest.cpp vec2ang.cpp )
