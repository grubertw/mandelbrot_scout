# Mandelbrot Scout
Renders the Mandelbrot Set on the GPU using Rusts's WGPU library. Leverages perturbation theory to overcome the GPU's float 32 precision limitation, and instead seeking to resolve qualified reference orbits that are computed on the CPU.  

The state of the repo is still VERY much a work-in-progress, as I make headway on the 'ScoutEngine' concept for CPU reference orbit discovery, computation, and utilization by the GPU-resident perturbance algorithm. 

I always try to keep HEAD of the repo tested and running, even if/when experimental features are still being developed.

## To compile:
$> cargo build

## To build and run:
$> cargo run

**If you don't have Rust on your system yet, it is very easy to install! Simply download the rustup shell script from [rust-lang.org](https://rust-lang.org/tools/install/), install rust, and then run `cargo run` from the shell once inside the project directory (where Cargo.toml is located). Cargo downloads and builds all the library dependencies, then the project, and then runs! Rust is also very portable, and works perfectly fine, right on Windows! While the shell script installer is meant for posix systems, the Rust community reccomends Chocolaty**

## Background
This project started out with my desire to learn more about OpenGL and how shaders work. As a lover of fractals, I had come across lots of articles that mentioned how the beautifully simple Mandelbrot algorithm can be parallelized. Ideally, each pixel - which can be mapped to a logical coordinate on the complex plane - can calculate its corresponding orbit - i.e. iteration steps until the coordinate escapes with a magnitude greater than 2 - as a completely independent operation. The only info that's needed is the number of iterations until escape, which is then used to compute a color. Well, what better way to compute colors per pixel than on a GPU, whose hardware was built for such a purpose? While some examples of rendering a fractal this way were around when I began looking (I started this in 2017, lol), they were all using GLSL and interfaced with OpenGl through C/C++. I wanted to use Rust though, and thought this could be a great way to learn that language, along with some newer graphics libraries that were making their way into the (at the time VERY new) Rust ecosystem.

A tremendous amount has changed, of course, since I started this, and Rust is making its way onto the scene in a big way, especially for game development and 3d graphics applications. When I switched to the Iced GUI library, I wasn't so sure it would last, but that seemed to be a good decision, and it was one of the few graphics libraries where I could find examples where I could overlay UI widgets on top of a GL canvas - like inside a video game. The Iced project ended up doing far more for me than just UI though. As I continued to search for ways to zoom deeper into the fractal, I came to understand both the benefits, and unfortunatly also the limitations, of WGPU and WGSL. My limitations with WGSL however are not unique, I later discovered, as there really is no such thing as a shader language that runs on the GPU with strong support for double precision. Still, despite this shortcomming, the GPU remains hand's down the best choice for Mandelbrot computation - and at least for the 'dumb' algorithm, is 'embarrasingly parallel'.  

# Project Goals
1. Perturbation theory & Series Approximation for deep zooms [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Perturbation_theory_and_series_approximation)
2. Ever increasing robustness and tuning for CPU-resident ScoutEngine, which is the 'brain' that makes perturbation calcuations on the GPU possible!
3. UI controls for ScoutEngine behavior
4. Coloring algorithms that leverage distance approximation 
5. UI-driven color palate selection 
6. Julia sets & cubic Mandelbrot 

The concept of ScoutEngine - which I have been iterating on with ChatGPT - is where my focus will remain on the project for some time to come. The concept is **big**, so I decided to rename my project to reflect that!

For anyone who has studied the ideas and concepts of Perturbation Theory for the Mandelbrot Set, the problem is non-trivial. Ultimately, it comes down to computing and then providing a very high precision reference orbit, i.e. a vector of complex numbers that represent Z-0 to Z-n, which can then be used for subsequent mandelbrot iterations, for 'nearby' pixels. I quote 'nearby' here very specifically, because even if two complex C for the fractal *are* close-by on the complex plane, that doesn't necessarily mean the reference orbit will be good!

This is where ScoutEngine comes in! While a researcher might spend some time to carefully select a good reference orbit, and one that might be applicable to every pixel in the viewport, the calculation then becomes extremely fragile to any slight changes to the viewport center and scale. Furthermore, a 'simple' perturbation calculation - i.e. one that linearizes the changes that occur to DeltaZ-n and ignores the higer-order term - remains highly sensitive to the quality of its reference orbit. In other words, not only does the pixel have to be (relatively) close to the reference orbit in the complex plain, but also the iterations them-selves *must not break* the rules of perturbation - i.e. all terms in the equation `∆n+1 = 2\*Zn\*∆n + ∆n^2 + ∆0` *must* remain small. 

What then is to be done, if picking a good reference orbit for nearby pixels is so hard? Well... a lot! ScoutEngine must be made not only to provide reference orbits, but good ones! It also has to provide a *lot* of them, especially at shallow zooms, where not only the differences in complex C per-pixel are large, but the geometry of the fractal is varying drastically. While there can still be lots of variation at deep-zoom, less reference orbits tend to be needed as the viewport window sees less hyperbolic components. In both cases, choosing where to 'seed' the reference orbit must first be regional, but choosing where and when the orbit is used, that must be heuristic. Heuristics must also ultimately drive the removal of orbits, so newer and more widely applicable orbits can be discovered.

ScoutEngine therefore uses a weighted set of indicies to score and rank individual orbits. Widely, this score can be used to cull orbits, and narrowly, it can be used choose a better one, even on a per-pixel basis. To be more useful per-pixel however, ScoutEngine must gather feedback from previous perturbation attempts, so that the score for the orbit may increase or decrease. It should also be noted that scoring from pertubation feedback (on the GPU) is more useful/applicable regionally, and ultimately becomes the best way for perturbation in general to succeed. 

# Screenshots
Old now, as plan on a far wider selection of color pallets and color algorithms.

![Mandelbrot 2.0 800x600 - screenshot 1](Mandelbrot_2_ss1.png)

### Other citations and tributes:
Here are the examples/tutorials I followed to help me get started:

https://github.com/iced-rs/iced/tree/0.13.1/examples/integration


NOTE: As with all examples in Git, make sure to view the correct code that has been tagged with the correct release of Iced. Also note, Iced seems to be good at keeping breaking changes from occurring within incremental releases; i.e. all iced_* libs in the 0.13.x series should be compatible. Using a newer version of wgpu might be possible, but only if its API updates are compatible with iced_wgpu; so it's probably not a good idea.

https://sotrh.github.io/learn-wgpu/#what-is-wgpu

Happy fractaling and happy coding!