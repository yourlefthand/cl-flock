# cl-flock
leverages opencl/pyopencl for large population flocking

## About

this project is my first experiment with opencl.
Goal is to implement a kernel workflow that can simulate individual behaviors in large groups with the intention of
creating a binary 'mask' image to select for certain experiences in photographic video of periodic or 'extracted' motion.

## TODO

not terribly efficient to be looping over the size of the viz retina inside the kernal.
perhaps use multiple kernals to pre-process into sub-sampled frames, then perform actions over those frames?

it would be nice to flesh out hte motion decision boundary to be more dynamic. Perhaps a perception-like sigmoid attached to a
4d non-negative vector?

simple gravitational acceleration & momentum physics (rather than desire = new-pos style movement) would make this look a lot
more natural and would go a long way toward simulating the 'murmuration' as desired:

https://www.youtube.com/watch?v=eakKfY5aHmY
