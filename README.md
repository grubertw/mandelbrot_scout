# Mandelbrot Scout
Renders the Mandelbrot Set on the GPU using Rusts's WGPU library and WGSL shaders, and Iced as the GUI overlay. Leverages perturbation theory to overcome the GPU's float 32 precision limitation, and instead discovers qualified reference orbits 'in the neighborhood', which are computed on the CPU in high precision.  

As a hobby project, the state of the repo is always a work-in-progress, especially for perturbation theory. That being said, I have made some very good progress lately, and can see quite visibly the drastic difference perturbation makes to overcome the GPU's precision wall! See in the screenshots section below!  

I always try to keep HEAD of the repo tested and running, even if/when experimental features are still being developed. As a stickler for quality UX, I will likely always remain hesitant to label 'official' releases on a regular basis.

## To compile:
`$> cargo build`

## To build and run:
`$> cargo run`

**If you don't have Rust on your system yet, it is very easy to install! Simply download the rustup shell script from [rust-lang.org](https://rust-lang.org/tools/install/), install rust, and then run `cargo run` from the shell once inside the project directory (where Cargo.toml is located). Cargo downloads and builds all the library dependencies, then the project, and then runs! Rust is also very portable, and works perfectly fine, right on Windows! While the shell script installer is meant for posix systems, the Rust community recommends Chocolaty.**

I'll also include here a special note about Rug, which I am using for MPFR and MPC. As a Linux user, I already had the GNU gcc toolchain installed on my PC (and rustup/cargo will take care of this for you on Linux/MacOS), but on Windows, this requires msys/mingw. Iced and WGPU both seek to be platform independent however, and I am not forcing any render back-end, or using any experimental GL shader features. On MacOS, Metal will likely be used, and on Windows, it will most likely be DirectX-12. Environment variables CAN be used to force a render back-end. 

Update (03/23/2026): Mandelbrot Scout v3.1 has been released successfully! I created a Windows VM with a properly configured environment to build the program! For me, what worked best was to use 'Git for Windows', and configure it's PATH - which is the Windows User PATH anyways - to include where MSYS installs mingw and it's binaries (when it's pacman command is run). This makes it so cargo can find gcc and use it to build MFPF. 

If you are trying to compile on windows and are running into issues, you can reach out to me through email, or on fractalforums.org, and I will try to help!  

## Background
This project started out with my desire to learn more about OpenGL and how shaders work. As a lover of fractals, I had come across lots of articles that mentioned how the beautifully simple Mandelbrot algorithm can be parallelized. Ideally, each pixel - which can be mapped to a logical coordinate on the complex plane - can calculate its corresponding orbit - i.e. iteration steps until the coordinate escapes with a magnitude greater than 2 - as a completely independent operation. The only info that's needed is the number of iterations until escape, which is then used to compute a color. Well, what better way to compute colors per pixel than on a GPU, whose hardware was built for such a purpose? While some examples of rendering a fractal this way were around when I began looking (I started this in 2017, lol), they were all using GLSL and interfaced with OpenGl through C/C++. I wanted to use Rust though, and thought this could be a great way to learn that language, along with some newer graphics libraries that were making their way into the (at the time VERY new) Rust ecosystem.

A tremendous amount has changed, of course, since I started this, and Rust is making its way onto the scene in a big way, especially for game development and 3d graphics applications. When I switched to the Iced GUI library, I wasn't so sure it would last, but that seemed to be a good decision, and it was one of the few graphics libraries where I could find examples where I could overlay UI widgets on top of a GL canvas - like inside a video game. The Iced project ended up doing far more for me than just UI though. As I continued to search for ways to zoom deeper into the fractal, I came to understand both the benefits, and unfortunately also the limitations, of WGPU and WGSL. My limitations with WGSL however are not unique, I later discovered, as there really is no such thing as a shader language that runs on the GPU with strong support for double precision. Still, despite this shortcomming, the GPU remains hand's down the best choice for Mandelbrot computation - and at least for the 'dumb' algorithm (i.e. absolute escape of `Zn=Zn^2+C`), is 'embarrassingly parallel'. Having the fragment shader handle escape-time iteration means that each pixel 'ideally' gets its own core! And with modern GPUs now having 10,000+ cores, that is far *far* more than the CPU will ever have!

That being said, my foremost goal with the project has always been: Make it fast! If I am not rendering the fractal in at least 25fps, then that's a fail! My aim is *always* to keep the UI super-fast and responsive (as long as you have a 'decent' graphics card), and not waiting for minutes (or hours even) for the fractal to render the viewport! Keeping that speed however, even with a powerful GPU, *does* start to become a serious challenge however, for even modest zooms, and when the iteration count must go higher. 

# Project Goals
1. Perturbation theory & Series Approximation for deep zooms [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Perturbation_theory_and_series_approximation)
2. Ever-increasing robustness and tuning for CPU-resident ScoutEngine, which finds and computes high-precision reference orbits for the scene/viewport!
3. UI controls for ScoutEngine behavior
   1. Basic controls are working! Right now it is best to control the Scout manually, with the 'Scout!' button. (See User Manual section below!)
4. Coloring algorithms that leverage distance approximation 
   1. Now using 4-neighbors strategy, after computing the distance derivative inside the Mandelbrot iteration loop!
5. UI-driven color palette selection 
   1. Color palettes are IN, and now configurable in settings.toml!
   2. Import MAP palettes also IN!
6. Julia sets & cubic Mandelbrot 

I've been iterating with ChatGPT on the 'Scout Engine' concept for a while now, and sometimes, if I am being honest, I wonder what the heck I was thinking to trust the AI to help me with design. It took me down some frightenting over-enginering paths - and this probably happened because of my own lack of understanding of perturbance - which created a hot-soup of complexity that I should have questened earlier on. That being said, it did help me understand the math better, and was useful for pouring through 10,000+ line log output.

For anyone who has studied the ideas and concepts of Perturbation Theory for the Mandelbrot Set, while the idea is simple, the solution is non-trivial. Fundamentally, it comes down to computing and providing a high precision reference orbit, i.e. a vector of complex numbers that represent Z-0 to Z-n, which can then be used for subsequent mandelbrot iterations, for 'nearby' pixels. I quote 'nearby' here very specifically, because even if two complex C for the fractal *are* close-by on the complex plane, that doesn't necessarily mean the reference orbit will be good!

Where I have most decidedly spent the most time - and where ChatGPT has plagued me - is "where" ecactly to spawn the reference orbit. This is where I probably should have spent more time studying how others approached this problem, rather than just blindly trusting the AI, but 'live and learn', I suppose. With that being said however, and with my biggest design goal to keep the render running FAST and responsive to the user, I needed an expediant way to both *find* and *spawn* a 'good enough' reference orbit that can be used for perturbation. Only after some trial and error did I discover that you CAN start with a reference in the Period-1 main cardoid, and render the *entire* set from that single reference! Don't expect deep zoom to be supported with an orbit so far from pixels, but shockingly (at least it was shocking to me!), it *does* work! 

My most recent re-factor of ScoutEngine now uses GPU feedback that has been reduced from a compute shader to help find 'good' reference orbits. My thought was: The GPU is already computing per-pixel escape during absolute, so why NOT use that as a hint to spawn seeds that have a high chance of being in the Set? While I have now taken a look at a few other approaches that use Newton's method, I still think there is merit to starting the search with some GPU hints (about it's per-pixel escape time calculations). The GPU does, after all, bring massive parallelism to the table, and can potentially allow 1000+ GPU cores to calculate escape. And from what I've seen so far, having a reference that can go at least say 1.5-2x past the iterations of perturbed pixels, then spawning a reference where the GPU indicates, works pretty well! And it's fast!

# User Manual
While the GUI is still being heavily worked-on - and is highly subject to change. As such, this might not be the most up-to-date, as I push changes.
1) Iteration Controls
   1) Max User (GPU) iters is now controlled by a slider
   2) There are text boxes now to change the step-size of the slider, and it's min/max range.
2) Color Controls
   1) Color Palette PickList
      1) Choose a color palette to use
      2) Starting Color palettes are loaded from the disk! All palettes (along with other program settings) are in settings/settings.toml
         1) New palettes using MAP file format can be added on-the-fly with the 'Import' button, which opens a file browser dialog. The new MAP palette will be added to color palette pick-list. 
   2) Palette Cycles slider
      1) A cycle of 1.0 means that no repetition of palette colors occurs. For the default 3-color R-G-B palette, these 3 colors would stretch across the (normalized) iteration count (t).
         1) The best way to think about this is normalized fractal iter count 't' against normalized color count 'c' - i.e. `t = iter / max_iter` & `c = c_iter / palette_len`.
   3) Offset slider
      1) Shift fractal iterations across the color palette. This value is always bounded between 0-1, and is relative to Palette Cycles.
   4) Gamma slider
      1) Applies a non-linear function (Pow) *after* a color has been picked from the palette - i.e. to the vec3 RGB value used directly by the GPU rasterizer.
   5) Color Scalar Mapping
      1) Choose between mapping functions applied to normalized fractal iteration count before the palette color is selected
         1) Choices are: Linear, Pow, Log, and Atan
      2) Mapping Strength allows the user to control the 'k' variable in the mapping functions - i.e `t = pow(t, k)` | `t = log(1.0 + k * t) / log(1.0 + k)` | `t = atan(k * t) / atan(k)`
         1) Note, the ranges of this value change depending on which mapping function is selected, allowing for the user to pick 'sane' values for functions, and affect the most aesthetically pleasing band of iteration counts with an appropriate change of color. 
   6) Distance Estimation controls 
      1) There are a LOT of additions here, and another complete overhaul of the UI. The best place to look for docs is in `settings.toml`
3) Scout Controls
   1) "Reset Scout" button will delete from program memory all reference orbits, and stop perturbation mode
      1) i.e. Absolute iteration will resume, with no reference orbit being used.
   2) Reference Iters Multiplier
      1) For the calculation of reference orbits, the max iterations used by the GPU will be multiplied using this value. Pertubation prefers orbits 'at least' as long as user/pixel max, but is better that it goes a little further. It also helps for reference orbit qualification, so this way the Scout can rank 'deeper' orbits before supplying them to the GPU.
   3) Spawn Count per Evaluation
      1) ScoutEngine uses 'evaluation cycles' to evaluate reference orbits. This value represents the number of reference orbits it will span from GPU-supplied seed values. If there are at least 'some' pixels 'in the set', then the reduce/compute shader will find then, and the Scout will seed from the translated coordinates. If the viewport is in filaments, then this value should be set higher, as it could take more cycles to find a good reference orbit that has the highest escape.
   4) Max Reference Orbits
      1) The maximum number of Reference Orbits the Scout will 'qualify' - i.e. send to the GPU for perturbing. 
   5) "Scout!" button will start the engine, entering perturbance render mode!
      1) At the moment, pertubation is a 'manual' process. However, the Scout was initially designed for automated orbit discovery, no matter where the user viewport window is located! Still, I think there may be a permanent use-case for running the Scout manually, and only launching evaluation cycles when clicking this button. As Reference Orbit computation is expensive, if the user hasn't noticed artifacts yet, then they may desire to keep the reference, rather than having the engine 'fight' with them anytime they pan/zoom the viewport camera. The user can also decide when/if they want to 'mine' the area for a bit longer, hunting for a better reference orbit that can allow them to pan/zoom with better clarity. This is also when it's worth increasing the 'Ref Iters Multiplier', as the deeper the orbit you can find in the area, the better! This makes orbit-finding a fully interactive process, and you can instantly see the visual difference when a better orbit is found/chosen/used! Also, a helpful diagnostics message will display in the top-left corner of the screen, informing the user of the Scout's activity!
4) Coordinate/Scale Controls
   1) "Poll" button will poll the GPU for its current viewport center and scale.
      1) When this button is clicked, then the three text-boxes for real, imaginary, and scale will populate. The user is then free to enter values of their choosing here, or 'save them' - i.e. by clicking the "Apply" button after moving the viewport, which will reset the GPU back to saved coordinates. Note the user CAN enter values into the text-boxes when/if they are empty, and then click apply - i.e. if you copy-paste some well-known coordinates from an internet source!
   2) Enter the Real complex coordinate for desired viewport center
   3) Enter the Imaginary complex coordinate for the desired viewport center
   4) Enter the desired viewport scale. 
   5) "Apply" button will take the values in the three text-boxes, and apply them to the GPU.
      1) Note that either scientific notation or decimal notation can be used inside these boxes. String validation of these values does not occur until the Apply button is clicked.
   6) "Restore From PNG" the scene from a PNG created by Mandelbrot Scout!
      1) See the section below on Saved Images
5) Save Controls
   1) Export the current scene/viewport as a PNG or JPEG
   2) If PNG is selected, the user may choose between None, Default, Fast, and Best compression.
      1) Default compression is what PNGs are typically saved with, and is considered a good balance.
      2) Fast is the lightest compression
      3) Best is the heaviest compression and will make the smallest file (for PNG), but comes with the cost of being slow!
   3) If JPEG is selected, the user is provided with a quality slider with values between 10 and 100 percent.
   4) If 'Use Alternate Dimensions' is checked, the user may enter an alternate resolution to save the image. 
      1) When using this feature, the user should pay careful attention to the current complex center and scale. Note that scale in the program is the complex space 'per-pixel', and NOT the viewport span - as is found in some fractal programs. For image export, this means that when the resolution is changed, more of the complex space can be seen, and effectively increases how much of the fractal appears in the image - i.e. the image will show more of the fractal that is currently cut-off by the viewport window. 
6) Resolution Controls
   1) "Resolution Factor" - i.e. 'RF' of 1.0 means that the fractal compute shader samples at pixel resolution, so if the viewport window is 1200/900, this means 1,080,000 fractal (mandelbrot) calculations are performed, per frame. 
      1) Using a value less-than 1.0 will apply the value as a multiplier to the current width and height of the viewport window, and effectively lower the resolution/quality of the image. WGPUs 'mag_filter' is then used to perform a linear blend/mix on render_tex, which the color shader writes. i.e. linearized subsampling. Using sub-one values here isn't really recommended, unless your GPU is really struggling, and you still want to keep a large window size. 
      2) Using a value greater-than 1.0 is where things get interesting! This is effectively supersampling, as more samples are taken than pixels in the viewport. with a value of 2.0, 4 samples are taken for every pixel. Here, WGPUs 'min_filter' within the texture sampler will perform a linear interpolation on these 4 values (which are stored in the texture-2d as rgb8unorm).
   2) "RF During Pan" is the resolution factor to use during pan operation. 
      1) It is especially handy to use values lower than 1.0 here if GPU appears sluggish, and can help a lot to increase responsiveness of the application. Using values greater than one make no sense here, as that would be supersampling during pan!
   3) "Samples" - Indicates the number of samples taken for a pixel-c value directly in the computation stage. This value is meant to be used with Jitter, and when both values are set greater than one, more fractal computations are performed, at jittered offsets from the pixel's exact center. When set to 4, for instance, then mandelbrot() will be called 4 times, at 4 different jittered offsets. These 4 samples will then be averaged together so that the color shader stage still only sees one (set) of fractal computations, when choosing the final color.
      1) In essence, ResFactor can be used to average color, and samples can be used to average the arithmetic (iters) calculations. Using all these sliders in the "Res" tab, you can effectively combine these averaging methods! It should be noted though that image export does not use the texture sampler, as it directly takes from the output color/render texture to PNG encode.  
   4) "Jitter" - the amount of Jitter applied to normalized pixel-c, where +-0.5 is the pixel's boundary. Keeping this in mind, applying a jitter strength of less 0.5 will keep samples 'inside' the pixel's 'box', and using values greater than 0.5 will force pixel samples outside this box, which effectively blurs/smooths the image!
      1) Note that pixel jitter is computed using a 128x128 PRNG (ChaCha8) white-noise texture, and is NOT blue noise (i.e. Poisson Disk). In later implementations, I may improve on this, and migrate to using true blue noise, and a geometric median for sample averages.
   5) "Averaging Bias" - Rather than take a basic average of iterations across samples, a mix() invocation is used. The bias is then used to mix between min_iters and max_iters found across samples.
      1) Effectively, this slider acts as sharpness. With Jitter and Bias both set to 1.0, the shader more aggressively includes pixels "into the set", making the image appear "sharper". Conversely, with Jitter left at 1 and Bias at 0, the set and it's filaments become "thinner". Ultimately, this is NOT sharpness, and depending on the color palette use, and how colors are mapped across iterations, using a bias of 0.0 may seem to "sharpen" the image - which was why I decided to make a slider for this value, as opposite extremes of this value may better suit the current color mapping. 

## Helpful environment variables to use on CLI while running
The RUST_LOG environment variable controls all the logging output, which is essential for debugging. Also, many other library dependencies (critically, WGPU) use the Rust logger, so this turns into the most essential environment variable for debugging!

Here are a few presets that I often use:

`$> export RUST_LOG=mandelbrot=trace` - This writes complete debug info for my program to STDERR.

`$> export RUST_LOG=wgpu_core=trace` - If you are having problems running the program - i.e. the window isn't opening, or it can't find your graphics card or a render back-end, use this!

`$> export RUST_LOG=mandelbrot::scout_engine=trace` - If you want to take a look at how the Scout is choosing reference orbits for the active viewport window!

`$> export WGPU_BACKEND=vulkan` - Force a render-backend. Valid strings are: `vulkan`, `metal`, `dx12`, or `gl`.
If you want to know more about what WGPU can support, look [here](https://github.com/gfx-rs/wgpu)

# Basic Use
I made this GIF to show how easy this program is to use!

Concerning settings.toml, while in v3.1, it was a hard requirement to have this file (and in a directory alongside the program called 'settings'), in versions 3.2 and further, it will no longer be necessary. That being said, this file is still useful for supplying overrides to the program, and making changes to its small set of initial color palettes. The file can now be placed directly next to the EXE, or put in a 'settings' dir (similar to repo structure), or you can also set a SETTINGS_DIR environment variable. If the program starts without this file however, there will only be one color palette available, the default RGB palette.

![Basic Use](screenshots/basic_use.gif)

# Image Export
Starting in version 3.2, you can now export/save the current Scene as a PNG or JPEG image. If PNG is selected, the program automatically writes a JSON file header. Information in this header can then be used to restore the Scene to the same complex location and scale! If you want to see the JSON text that is written to the PNG header, there is a handy little program you can use called 'pngcheck'. Here is an example of it's use and output (on Linux):

```
 ➜  pngcheck -t fractal.png
File: fractal.png (205372 bytes)
FractalMetadata:
    {"program_name":"Mandelbrot Scout","version":"1","center_re":"-9.864719622747442105567579431875481177930e-3","center_im":"1.029624344701343542604038494832493881498","scale":"1.232924115364685679889029198474463259543e-7","max_iter":500,"ref_orbit":null}
OK: fractal.png (1385x1000, 32-bit RGB+alpha, non-interlaced, static, 96.3%).
```

# Screenshots (Latest)
Perturbation Theory still stands as the greatest contributor to quality on the GPU. Not even super-sampling can help with the fundamental problem that float-32 values cannot carry enough precision to avoid the rounding error that occurs at scales lower than 1e-7 (i.e. forcing the representation between pixel-c values into less than 4 bits). With a good reference orbit, perturbation math comes to save the day, restoring how GPU f32 values can represent pixel-c - i.e. by detas to the reference orbit, rather than the absolute pixel-c values themselves!
![Pixelation at f32 precision boundary](screenshots/Screenshot_2026-04-11_11-52-59.png)

Below you can see that supersampling without perturbation doesn't help much, and that is because we lack enough bits (significant digits) to distinguish the difference between pixel-c 'boxes' (which are painfully visible in this render).
![Pixelated supersampling](screenshots/Screenshot_2026-04-12_06-53-52.png)

Here is what these exact same coordinates and scale look like after the "Scout!" button is pressed! With a single reference orbit that is 625 iterations in length - which is slightly longer than max_iters rendered by the GPU - we can faithfully approximate iterations for pixel-c based on delta-c and the reference orbit.
![Scout solution](screenshots/Screenshot_2026-04-11_11-53-41.png)

From here, there is still more we can do to enhance the quality of the image (mathematically), and that is super-sampling, which I have been working on and introduce with version 3.2! Perturbation Theory now gives us plenty of breathing room for adding samples in 'boxes' of pixel-c, which for this current scene, is a span of 5.4e-10. The first form of super-sampling I used leveraged a texture sampler and an "oversized" render texture, which I make independent of screen resolution, meaning that I can write more image data than pixels on the screen. The texture sampler gives a hardware based approach to perform liner interpolation of rgba8 integer values for the case of minification. The texture sampler also uses mip-mapping, which does provide some benefit for fractals, but is far better suited to anti-alias 3d images.
![Super-sampling using Texture Sampler](screenshots/Screenshot_2026-04-12_07-01-15.png)
Looking closely, you can notice less 'artifacts' in the image - i.e. the black pixels that are close but not inside the minibrot, which are not actually 'in the set'. The texture sampler here averages/smooths the colors for us, resulting in a cleaner aesthetic!

A better form of supersampling is to more tightly control where our iterations from pixel center occur. With the "pure resolution" based supersampling above, "sub-pixel-c" values become laid out in a grid-like pattern (think of graph paper), and still doesn't solve well the occurrence of 'Moiré patterns'. Rather than relying on a simple scale factor of the sampling grid, a better way is to use jittered offsets from pixel-c. Calculating the angle and distance of the jittered pixel can be tricky, and there are lots of different ways to do this. The approach I decided to take (at least for the screenshot below) is to use a procedurally generated noise texture, i.e. u8 values generated by the ChaCha8 PRNG algorithm. The 'text book' ideal is to use a blue-noise texture, but with a careful selection of values from this texture, we still easily avoid the problem of correlation artifacts (which is the primary purpose of using noise for jitter offsets). The other important difference is how samples are averaged. If only color is averaged, this hides iteration counts, which is the best indicator for how samples can best be combined. After fiddling with a few different averaging algorithms, I had the thought to introduce yet another slider, which is the "Averaging Bias". 

![Super-sampling with max_iters bias](screenshots/Screenshot_2026-04-12_07-13-44.png)
At least for my debug coloring - which is purely algorithmic: `color = vec3f(t, t*t, pow(t, 0.5))` where `t = it / max_it` (i.e. normalized iterations count) - taking a high sample count (i.e. 16 samples per pixel-c) - and then applying jitter with a strength of 1.0 (which forces some samples taken outside the pixel's 'box' (any jitter > 0.5 will do this)) - and with an averaging bias of 1.0, which effectively changes the average to a max of iter counts for all samples taken, we get the above (maybe a bit too aggressively) sharpened image! 

Below is when we switch the Averaging bias to 0 instead, which translates to the min of iter counts for all samples.
![Super-sampling with min_iters bias](screenshots/Screenshot_2026-04-12_07-31-02.png)

Surprisingly, I find myself liking a bias toward min_iters for most 256-color MAP palettes that find popular use on fractalforums, as it does a better job of smoothing what would otherwise be (perhaps) too busy level of detail. 
![FarDareisMaiRainbows](screenshots/Screenshot_2026-04-12_08-45-55.png)

![FarDareisMaiPainting](screenshots/Screenshot_2026-04-12_08-47-56.png)

![FarDareisMaiPainting max_iters](screenshots/Screenshot_2026-04-12_08-48-36.png)

And the final PNG export.
![FarDareisMayPainting export](screenshots/screenshot1.png)

Ultimately, it all comes down to finding the right aesthetic appeal, which is why I like sliders so much. With a good range of values, these sliders give users highly interactive and highly responsive feedback, especially when making very small changes to values, which would be nearly impossible to see the difference otherwise. Rendering on the GPU is also what makes these kinds of successive coloring+calculation changes even possible - at least at high frame-rates - and I feel like that matters a lot for these types of fine-tuning.   

## 3.1 Update - Distance Estimation is working now!

![De demo 1](screenshots/de_demo.gif)

Here are a few clean screenshots, as the moving gifs don't show high quality.

![De screenshot 1](screenshots/Screenshot_2026-03-15_23-14-51.jpg)

![De screenshot 2](screenshots/Screenshot_2026-03-15_23-16-37.jpg)

![De screenshot 3](screenshots/Screenshot_2026-03-15_23-18-59.jpg)

Here are a dew DE renders that are working with perturbation!

![De Pertrub Screenshot 1](screenshots/Screenshot_2026-03-15_23-35-16.jpg)

![De Perturb Screenshot 2](screenshots/Screenshot_2026-03-15_23-36-07.jpg)

![De Perturb Screenshot 3](screenshots/Screenshot_2026-03-15_23-37-28.jpg)

Here are a rew renders that show stripe averaging!

![Stripe De Perturb 1](screenshots/Screenshot_2026-03-16_00-05-08.jpg)

![Stripe De Perturb 2](screenshots/Screenshot_2026-03-16_00-21-36.jpg)

## Older screenshots
Here is some craziness that ChatGPT was having me do while attempting to get pertubation working. It had me hunting for orbits with a perfect 'r_valid' for linerized perturbation (which is NOT necessary when you keep the quadratic term). What did come out of this effort however, was a highly scalable light-weight threading model, which allowed me to spawn and compute hunderads of reference orbits per second (orbits were short here, all under 8192 iterations), accross all 16 of my CPU cores! In the render below, I had over 250 independant worker tasks - wrapped as async functions - which were all enqued into a thread-pool with non-blocking awaits for orbit computation results, and feeding to the GPU/main thread to flow into the GPU render pipeline after results were ready.
![Older ScoutEngine failure](screenshots/Screenshot_2026-02-25_10-41-24.png)

Here are a few more 'before and after pertubation snapshots, to show how much it shifts the precision wall!
![Precision Wall 1](screenshots/Screenshot_2026-03-04_15-27-26.png)
![Precision Wall 1 Fixed](screenshots/Screenshot_2026-03-04_15-27-35.png)
![Precision Wall 2](screenshots/Screenshot_2026-03-04_15-28-59.png)
![Precision Wall 2 Fixed](screenshots/Screenshot_2026-03-04_15-29-06.png)

A few more nice ones that I have been able to make at zoom past typical GPU shader support!
![Seahorse minibrot 7e-10](screenshots/Screenshot_2026-03-04_15-36-21.png)
![Seahorse mini-mini 3e-14](screenshots/Screenshot_2026-03-04_15-43-32.png)

Some preliminary deep-zoom tests...
![Deep Zoom Prelim 1](screenshots/Screenshot_2026-03-04_10-15-01.png)
![Deep Zoom Prelim 2](screenshots/Screenshot_2026-03-04_15-40-39.png)
![Deep Zoom Prelim 3](screenshots/Screenshot_2026-03-04_16-23-08.png)

## Screenshots from 2.0
Here is my oldest screenshot, where I initially had sliders to control RGB based on a sin wave, and then modulating the frequency of the wave, relative to escape time.

![Mandelbrot 2.0 800x600 - screenshot 1](screenshots/Mandelbrot_2_ss1.png)

### Other citations and tributes:
Here are the examples/tutorials I followed to help me get started:

https://github.com/iced-rs/iced/tree/0.14.0/examples/integration

NOTE: As with all examples in Git, make sure to view the correct code that has been tagged with the correct release of Iced. Also note, Iced seems to be good at keeping breaking changes from occurring within incremental releases; i.e. all iced_* libs in the 0.14.x series should be compatible. Using a newer version of wgpu might be possible, but only if its API updates are compatible with iced_wgpu; so it's probably not a good idea.

https://sotrh.github.io/learn-wgpu/#what-is-wgpu

# License 
The MIT License (MIT)

Copyright © 2026 Travis Gruber

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.