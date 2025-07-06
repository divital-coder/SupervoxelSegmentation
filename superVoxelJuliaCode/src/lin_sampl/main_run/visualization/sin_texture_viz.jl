"""
When training using sinusoid loss we have texture representation for each tetrahedron  
    we want to display it so 
    1) we get triangles and associated supervoxel indexes from get poligon files
    2) we prepare each time we display separate fragment shader that will iterate ove texture bank 
    3) we load the sinusoid texture - we flatten it and access necessery element to get the offset and importance of each texture in texture bank
    4) we remember to use the image coordinates so we will have offses properly used and each traingle properly displaying texture 
"""
```
You are opengl expert. Goal is to create fragment shader source code that would be able to display intensity values of each triangle depending on this triangle index .  Main equation is "sin(2 * π / p[4] * ((x+p[5]) * cos(p[1]) + (y+p[6]) * cos(p[2]) + (z+p[7]) * cos(p[3])))+p[8]" where z,y should be calculated on the basis of opengl coordinates in the image , z is constant and given   """            alpha=p[1]*2*π
            beta=p[2]*2*π
            gamma=p[3]*2*π
            wavelength=p[4]*max_wavelength
            offset_x=p[5]*max_wavelength
            offset_y=p[6]*max_wavelength
            offset_z=p[7]*max_wavelength
            base=p[8] *max_amplitude
            amplitude=p[9]*max_amplitude""" 
Code of fragment shader should be generated based on  "texture_bank_p" tensor . it is 3 dimensional tensor where first dimension is bank index second sinusoid index and third is for sinusoid values - has length 5 and 1) alpha 2) beta 3) gamma 4) wavelength 5) amplitude.
Morover data would be loaded via vertex shader with each point of the triangles based on "sin_p" tensor. "sin_p" has 2 dimensions fist is triangle index, and in second first 5 entries are  1) offset_x, 2) offset_y, 3) offset_z  4)bias 5) multiplier and from entry 6 on we have 1 entry for each texture bank marking the influence of given texture bank on final value (all influences sum to 1) . 
Summarizing given texture_bank_p create code that prepare code for fragment shader and given sin_p prepare code that prepare vertex shader (values from sin_p needs to be passed per triangle)
    ```