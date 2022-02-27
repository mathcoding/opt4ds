\* Source Pyomo model name=unknown *\

min 
x8:
+1 x1
+1 x2
+1.01 x3
+1.02 x4
+1.03 x5
+1.0700000000000001 x6
+1.05 x7

s.t.

c_l_x9_:
+1 x1
+1 x4
+1 x5
+1 x6
+1 x7
>= 5

c_l_x10_:
+1 x1
+1 x2
+1 x5
+1 x6
+1 x7
>= 6

c_l_x11_:
+1 x1
+1 x2
+1 x3
+1 x6
+1 x7
>= 7

c_l_x12_:
+1 x1
+1 x2
+1 x3
+1 x4
+1 x7
>= 3

c_l_x13_:
+1 x1
+1 x2
+1 x3
+1 x4
+1 x5
>= 3

c_l_x14_:
+1 x2
+1 x3
+1 x4
+1 x5
+1 x6
>= 2

c_l_x15_:
+1 x3
+1 x4
+1 x5
+1 x6
+1 x7
>= 2

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
   0 <= x3 <= +inf
   0 <= x4 <= +inf
   0 <= x5 <= +inf
   0 <= x6 <= +inf
   0 <= x7 <= +inf
general
  x1
  x2
  x3
  x4
  x5
  x6
  x7
end
