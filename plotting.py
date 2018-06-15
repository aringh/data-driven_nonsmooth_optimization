"""Code for creating the plot in the paper"""

import numpy as np
import matplotlib.pyplot as plt

save_path = 'Give a save path'


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
iter_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 27, 30,
              33, 36, 39, 42, 45, 50, 55, 60, 70, 80, 90, 100, 150, 200, 250,
              300]

# Results as average over 100 phantoms
loss_sidky = [18939.502158203126, 5025.5223461914065, 2067.0353845214845,
              1031.4667388916016, 623.06912780761718, 466.60300750732421,
              400.4218392944336, 361.47143035888672, 331.44562438964846,
              307.17602478027345, 288.30622772216799, 274.07745544433595,
              248.31353317260744, 234.24225967407227, 225.42493766784668,
              219.48006118774413, 215.23997070312501, 212.09245178222656,
              209.68343795776366, 207.79406005859374, 206.28261077880859,
              205.05378005981444, 204.04111808776855, 202.71044845581054,
              201.70333206176758, 200.92532341003417, 199.82694763183594,
              199.11397659301758, 198.63221626281739, 198.29623893737792,
              197.57292976379395, 197.37533462524414, 197.30535522460937,
              197.27619499206543]

loss_cp_const = [12886.935864257812, 3510.6859374999999, 1399.0583532714843,
                 659.00321716308599, 468.40173004150392, 438.02103546142575,
                 410.64037628173827, 363.5603674316406, 315.28353790283205,
                 279.62799102783202, 257.87679611206056, 245.26353256225585,
                 225.95082260131835, 216.28674453735351, 211.133060836792,
                 207.79802726745606, 205.5376668548584, 203.91241615295411,
                 202.69393508911133, 201.75682289123534, 201.02152137756349,
                 200.43507926940919, 199.96114624023437, 199.35437835693358,
                 198.91007316589355, 198.57754737854003, 198.12849472045897,
                 197.85297943115233, 197.67618385314941, 197.55838035583497,
                 197.32764778137206, 197.2742194366455, 197.25765716552735,
                 197.25151260375978]

loss_our_2x2 = [180977.49460937499, 103398.1903515625, 40974.235830078127,
                12934.823881835937, 3311.2684631347656, 877.31835174560547,
                369.17604156494139, 244.1925839996338, 222.44221405029296,
                221.42951957702635, 215.50117004394531, 210.15300270080567,
                205.31969169616698, 202.7324617767334, 201.18980865478517,
                200.17820907592773, 199.47986991882325, 198.98071517944337,
                198.61470878601074, 198.34082000732423, 198.13216163635255,
                197.97064826965331, 197.84404045104981, 197.68805557250977,
                197.57898353576661, 197.50089736938477, 197.40151451110839,
                197.34509552001953, 197.31142684936523, 197.29050506591796,
                197.25502494812011, 197.24878273010253, 197.2472582244873,
                197.24680015563965]

loss_cp = [177568.36874999999, 99800.789648437494, 39084.240976562498,
           12264.618803710937, 3101.2784545898439, 846.67608215332029,
           388.93958206176757, 252.46717437744141, 223.79262588500976,
           225.00771293640136, 217.26661735534668, 209.11574607849121,
           204.27016532897949, 201.70856002807616, 200.35189834594726,
           199.51293144226074, 198.95216743469237, 198.55739852905273,
           198.27155799865722, 198.0597243499756, 197.89971351623535,
           197.77689697265626, 197.68139686584473, 197.56472198486327,
           197.48413764953614, 197.4272787475586, 197.35644210815428,
           197.31780197143556, 197.2958839416504, 197.28319534301758,
           197.26755088806152, 197.26978858947754, 197.27350715637206,
           197.27663574218749]

loss_general_nm2 = [25133.611562499998, 9579.276791992188, 3766.6393432617188,
                    1675.0945513916015, 614.80189483642573, 388.86414230346679,
                    280.45288146972655, 232.67581756591798, 221.72526504516603,
                    217.27104461669921, 212.78144134521483, 208.80965476989746,
                    205.57998359680175, 201.87361968994139, 201.15900695800781,
                    200.31115478515625, 200.61713027954102, 201.36368919372558,
                    203.17898078918458, 206.33866371154784, 211.85435333251954,
                    221.37628929138182, 238.23620193481446, 303.99950210571291,
                    512.95106170654299, 1298.7509350585938, 21942.769462890625,
                    622258.85062499996, 20314128.739999998, 698223412.15999997,
                    44445109018312048.0, 3.4943773972890217e+24, 3.0364593370338254e+32,
                    np.inf]

loss_general_nm3 = [5526.8552441406246, 6227.8524243164065, 928.92287109375002,
                    532.07267700195314, 366.97947570800784, 279.29114532470703,
                    245.24093276977538, 222.62817131042482, 217.06319129943847,
                    212.15167015075684, 210.64053039550782, 207.48350738525392,
                    209.38224647521972, 212.69040100097655, 279.17968887329101,
                    508.82947814941406, 2091.5332922363282, 13776.14638671875,
                    152989.184453125, 2302716.473125, 39822794.369999997,
                    724085302.24000001, 13443981265.92, 1791730730598.3999,
                    243351063992729.59, 33512791449177948.0, 6.5539290926932872e+20,
                    1.3205202060865098e+25, 2.7184337183860469e+29, 5.6880178123594228e+33,
                    np.inf, np.nan, np.nan, np.nan]


normalize_values = True
normalization_value = 197.246657104  # Computed using CP solver with Sidky et.al parameters, 1000 iter.

if normalize_values:
    loss_sidky = [value - normalization_value for value in loss_sidky]
    loss_cp_const = [value - normalization_value for value in loss_cp_const]
    loss_our_2x2 = [value - normalization_value for value in loss_our_2x2]
    loss_cp = [value - normalization_value for value in loss_cp]
    loss_general_nm2 = [value - normalization_value for value in loss_general_nm2]
    loss_general_nm3 = [value - normalization_value for value in loss_general_nm3]


# -----------------------------------------------------------------------------
# Create the plot
# -----------------------------------------------------------------------------
# Create the big plot
fig, ax1 = plt.subplots()

lines = ax1.plot(iter_count, loss_sidky)
plt.setp(lines, color='xkcd:red', linestyle='-', linewidth=1.5)

lines = ax1.plot(iter_count, loss_cp_const)
plt.setp(lines, color='xkcd:green', linestyle='-', linewidth=1.5)

lines = ax1.plot(iter_count, loss_our_2x2)
plt.setp(lines, color='xkcd:sky blue', linestyle='-', linewidth=1.5)


lines = ax1.plot(iter_count, loss_cp)
plt.setp(lines, color='xkcd:orange', linestyle='-.', linewidth=1.5)

lines = ax1.plot(iter_count, loss_general_nm2)
plt.setp(lines, color='xkcd:purple', linestyle='-.', linewidth=1.5)

lines = ax1.plot(iter_count, loss_general_nm3)
plt.setp(lines, color='xkcd:black', linestyle='-.', linewidth=1.5)


ax1.axis([1, 300, 0.01, 10000])

plt.xscale('log')
plt.yscale('log')

plt.axvline(x=10, linestyle='--',  color='xkcd:dark gray', linewidth=0.8)

plt.rc('font', family='serif')

ax1.legend(['(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)'])


# Create a small subplot
left, bottom, width, height = [0.2, 0.18, 0.25, 0.23]
ax2 = fig.add_axes([left, bottom, width, height])

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax1, ax2, loc1=1, loc2=2, joinstyle='round')

lower_index = 6
upper_index = 15

lines = ax2.plot(iter_count[lower_index:upper_index], loss_sidky[lower_index:upper_index])
plt.setp(lines, color='xkcd:red', linestyle='-', linewidth=1.5)

lines = ax2.plot(iter_count[lower_index:upper_index], loss_cp_const[lower_index:upper_index])
plt.setp(lines, color='xkcd:green', linestyle='-', linewidth=1.5)

lines = ax2.plot(iter_count[lower_index:upper_index], loss_our_2x2[lower_index:upper_index])
plt.setp(lines, color='xkcd:sky blue', linestyle='-', linewidth=1.5)


lines = ax2.plot(iter_count[lower_index:upper_index], loss_cp[lower_index:upper_index])
plt.setp(lines, color='xkcd:orange', linestyle='-.', linewidth=1.5)

lines = ax2.plot(iter_count[lower_index:upper_index], loss_general_nm2[lower_index:upper_index])
plt.setp(lines, color='xkcd:purple', linestyle='-.', linewidth=1.5)

lines = ax2.plot(iter_count[lower_index:upper_index], loss_general_nm3[lower_index:upper_index])
plt.setp(lines, color='xkcd:black', linestyle='-.', linewidth=1.5)

plt.xscale('log')
plt.yscale('log')

plt.axvline(x=10, linestyle='--',  color='xkcd:dark gray', linewidth=0.8)


ax2.axis([7, 20, 5, 100])
ax2.xaxis.set_ticks([1], minor='True')


# Save the figure
fig.savefig(filename=save_path+'obj_fun_values', format='eps')
