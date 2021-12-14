#!/usr/bin/env python
# coding: utf-8

# Supplementary codes for: 
# #Potential severity and control of Omicron waves depending on pre-existing immunity and immune evasion
# 
# Ferenc A. Bartha, Péter Boldog, Tamás Tekeli, Zsolt Vizi, Attila Dénes and Gergely Röst
# 
# 
# 
# ---

# ## Colab Configuration

# In[ ]:


use_colab = False

if use_colab:
  from google.colab import files


# ## Imports

# In[ ]:




from ipywidgets import interact

import matplotlib.pyplot as plt
from matplotlib import colors as c
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.font_manager as mfm

import numpy as np

from scipy.integrate import odeint 
import math


# ## Parametrization

# ### Epidemiological Parameters

# In[ ]:


# Delta variant

# basic reproduction number of the Delta variant 
#  (relevant for fully susceptible population with no interventions in place)
r0_delta = 6.0


# In[ ]:


# Observations from South Africa (Laboratory Country)

# ratio of the immunized population
p_south_africa = 0.85

# ratio of the effective reproduction numbers - as observed: 
#  R_t^{Omicron} / R_t^{Delta}
ratio_omicron_per_delta_south_africa = 4


# In[ ]:


# Assumptions on Omicron

# latent period (days): 2-chain        L1->L2
omicron_latent_period = 2.5

# infectious period (days): 4-chain    I1->I2->I3
omicron_infectious_period = 5.

# hospital evasion with pre-existing immunity (probability of evasion)
omicron_hospital_evasion = 0.85


# In[ ]:


# Deriving model parameters from the above assumptions

# alpha
alpha = 1. / omicron_latent_period

# gamma
gamma = 1. / omicron_infectious_period


# ### Technical Parameters

# In[ ]:


# Region for immune evasion (e) and local pre-existing immunity (p_loc)

# immune evasion (e)
e_vals = np.linspace(0, 1, 100)

# local pre-existing immunity (p_loc)
p_loc_vals = np.linspace(0, 1, 100)


# In[ ]:


# ODE solver

# integration timespan and resolution (t)
t = np.linspace(0, 500, 5000)

# Model compartments
comps = ["s", "l1_s", "l2_s", "i1_s", "i2_s", "i3_s", "i4_s", "r_s",          "p", "l1_p", "l2_p", "i1_p", "i2_p", "i3_p", "i4_p", "r_p"]


# In[ ]:


# Figures

# resolution
figures_dpi = 250

# auto download
figures_autodownload = True


# ## Methods

# ### Contour relation: pre-existing immunity vs immune evasion

# In[ ]:


def r0_omicron_from_contour_relation(
    e: float, 
    p: float = p_south_africa, 
    r0_delta: float = r0_delta, 
    ratio_omicron_per_delta: float = ratio_omicron_per_delta_south_africa) -> float:
  """
  Approximates the basic reproduction number (R0) of the Omicron variant
  :param float e: immune evasion of Omicron, i.e. ratio of individuals with 
                  immunity against Delta who are susceptible to Omicron
  :param float p: pre-existing immunized fraction of the population
  :param float r0_delta: basic reproduction number of the Delta variant
  :param float ratio_omicron_per_delta: ratio of effective reproduction numbers 
                                        for Omicron and Delta variants
  :return float: basic reproduction number of the Omicron variant
  """
  num = r0_delta * ratio_omicron_per_delta
  denom = 1 + (0 if p == 1 else e * p / (1 - p))

  return num / denom


# ### Level of non-pharmaceutical interventions (NPI) required to suppress an epidemic

# In[ ]:


def calculate_suppressing_npi(r0: float,
                              p: float,
                              goal: float = 1) -> float:
    """
    Calculate the necessary contact rate reduction to achieve the <goal> rep. number
    :param float r0: basic reproduction number
    :param float p: pre-existing immunity
    :param float goal: desired reproduction number (<= 1)
    :return float: NPI 
    """
    return 0 if (p == 1) else 1 - min(1, goal / (r0 * (1 - p)))


# ### Compartmental ODE modeling of the Omicron variant

# In[ ]:


def omicron_model(
    xs: np.ndarray, 
    ts: np.ndarray, 
    params: dict
  ) -> np.ndarray:
  """
  SL_2I_4R model with dual immunity
  :param np.ndarray xs: actual array of states
  :param np.ndarray ts: time values
  :param dict params: dictionary of parameters
  :return np.ndarray
  """
  # get parameters
  alpha = params["alpha"]
  beta  = params["beta"]
  gamma = params["gamma"]
  npi   = params["npi"]

  # get all states
  # _s: individuals susceptible to both Omicron and Delta
  # _p: individuals susceptible to Omicron but immune to Delta
  s, l1_s, l2_s, i1_s, i2_s, i3_s, i4_s, r_s,   p, l1_p, l2_p, i1_p, i2_p, i3_p, i4_p, r_p = xs

  # total count of infectious individuals
  i_sum = i1_s + i2_s + i3_s + i4_s + i1_p + i2_p + i3_p + i4_p

  # compartmental model
  ds    = - beta * (1 - npi) * s * i_sum
  dl1_s =   beta * (1 - npi) * s * i_sum - 2 * alpha * l1_s
  dl2_s = 2 * alpha * l1_s - 2 * alpha * l2_s 
  di1_s = 2 * alpha * l2_s - 4 * gamma * i1_s 
  di2_s = 4 * gamma * i1_s - 4 * gamma * i2_s 
  di3_s = 4 * gamma * i2_s - 4 * gamma * i3_s 
  di4_s = 4 * gamma * i3_s - 4 * gamma * i4_s 
  dr_s  = 4 * gamma * i4_s
  
  dp    = - beta * (1 - npi) * p * i_sum
  dl1_p =   beta * (1 - npi) * p * i_sum - 2 * alpha * l1_p
  dl2_p = 2 * alpha * l1_p - 2 * alpha * l2_p 
  di1_p = 2 * alpha * l2_p - 4 * gamma * i1_p 
  di2_p = 4 * gamma * i1_p - 4 * gamma * i2_p 
  di3_p = 4 * gamma * i2_p - 4 * gamma * i3_p 
  di4_p = 4 * gamma * i3_p - 4 * gamma * i4_p 
  dr_p  = 4 * gamma * i4_p
  
  return np.array([ds, dl1_s, dl2_s, di1_s, di2_s, di3_s, di4_s, dr_s,                    dp, dl1_p, dl2_p, di1_p, di2_p, di3_p, di4_p, dr_p])


# In[ ]:


def calculate_beta(
    s0: float, 
    r0: float,
    params: dict
  ) -> float:
  """
  Calculate beta from R0 and other parameters
  :param float s0: initial ratio of susceptibles
  :param float r0: basic reproduction number
  :param dict params: dictionary of parameters
  :return float: calculated beta
  """
  
  return r0 * params["gamma"]


# In[ ]:


def solve_omicron_model(
    r0_omicron: float, 
    e: float,
    p_loc: float, 
    npi_loc: float,
    initial_l1: float,
    t: np.ndarray = t
  ) -> list:
  """
  Calculate peak and final sizes
  :param float r0_omicron: basic reproduction number of the Omicron variant
  :param float e: immune evasion of Omicron
  :param float p_loc: pre-existing immunity in the model country
  :param float npi_loc: npi in effect in the model country
  :param float initial_l1: initially infected (L1_s + L1_p, symmetric)
  :param np.ndarray t: timespan and resolution of the numerical solution
  :return list: numerical solution to the omicron model
  """
  
  # initial values
  s_0    = 1 - p_loc
  l1_s_0 = initial_l1 / 2.
  l2_s_0 = 0.0
  i1_s_0 = 0.0
  i2_s_0 = 0.0
  i3_s_0 = 0.0
  i4_s_0 = 0.0
  r_s_0  = 0.0

  p_0    = e * p_loc
  l1_p_0 = initial_l1 / 2.
  l2_p_0 = 0.0
  i1_p_0 = 0.0
  i2_p_0 = 0.0
  i3_p_0 = 0.0
  i4_p_0 = 0.0
  r_p_0  = 0.0

  iv = [s_0, l1_s_0, l2_s_0, i1_s_0, i2_s_0, i3_s_0, i4_s_0, r_s_0,         p_0, l1_p_0, l2_p_0, i1_p_0, i2_p_0, i3_p_0, i4_p_0, r_p_0]

  # set readily known parameters
  params = {
      "alpha": alpha,
      "gamma": gamma,
      "npi": npi_loc
  }

  # calculate beta
  beta = calculate_beta(
      s0 = s_0 + p_0, 
      r0 = r0_omicron,
      params = params
  )

  params["beta"] = beta

  # compute the numerical solution
  sol = odeint(
      func = omicron_model, 
      y0 = iv, 
      t = t, 
      args = (params, )
  )

  return sol


# In[ ]:


def calculate_peak_and_final_size(
    sol, 
    severity: float = 1, 
    relative_severity: float = (1 - omicron_hospital_evasion)) -> list:
  """
  Calculate peak and final sizes
  :param ODESolution sol: solution of the numerical simulation
  :param float severity: common weight of _s and _p compartments
  :param float relative_severity: additional weight of _p compartments
  :return list: peak and final size
  """
 
  # unwrap the ODE solution
  sol_d = {comps[i]: sol[:, i] for i in range(len(comps))}

  # plug-in weights
  r = severity * (sol_d["r_s"] + relative_severity * sol_d["r_p"])

  i = severity * (
                           sol_d["i1_s"] + sol_d["i2_s"] + sol_d["i3_s"] + sol_d["i4_s"] + \
      relative_severity * (sol_d["i1_p"] + sol_d["i2_p"] + sol_d["i3_p"] + sol_d["i4_p"])
    )
    
  # peak size
  peak_size = np.max(i)

  # final size
  final_size = r[-1]

  return peak_size, final_size  


# ## Results

# ### Contours: R0 of Omicron vs immune evasion

# #### Code

# In[ ]:


def plot_r0_omicron_vs_immune_evasion(
    es,
    ps,
    save_this_figure = False
  ) -> None:
  """
  Plot R0 of Omicron depending on its immune evasion
  :param list es: immune evasion values for the horizontal axis (resultion)
  :param list ps: pre-existing immunity values (number of curves)
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """
  
  # ensure proper fontsize
  plt.rcParams.update({'font.size': 10})

  # setup the coloring scheme
  magic_color_count = round(1.5 * len(ps))
  colors = plt.cm.winter(np.linspace(0, 1, magic_color_count + 1))

  # setup the figure
  fig, ax = plt.subplots(
      dpi=figures_dpi if save_this_figure else 180, 
      figsize=(5, 3)
    )

  # plot a contour for each p \in ps
  for idx, p in enumerate(ps):

    r0_omicron_vals = r0_omicron_from_contour_relation(
        e = es,
        p = p,
        r0_delta = r0_delta, 
        ratio_omicron_per_delta = ratio_omicron_per_delta_south_africa
    )

    ax.plot(es, r0_omicron_vals, 
             label=str(round(p, 2)), 
             color=colors[magic_color_count - idx])
    
  lgd = ax.legend(loc='right', bbox_to_anchor=(1.6, 0.5), 
              title='Pre-existing immunity\nin South Africa\n(fraction of population)')

  ax.set_xlim(0, 1)
  ax.set_ylim(0, r0_delta * ratio_omicron_per_delta_south_africa)

  ax.set_yticks(range(0, int(r0_delta * ratio_omicron_per_delta_south_africa) + 1, 4))

  ax.set_xlabel('immune evasion')
  ax.set_ylabel('$R_0$ of Omicron')
  
  if save_this_figure == False:
    
    ax.set_title('Immune Evasion vs $R_0$ of Omicron')

  else:
    
    myFileName = "contourRelation.pdf"

    plt.savefig(myFileName, dpi=figures_dpi, 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# In[ ]:


def heatmap_r0_omicron_vs_immune_evasion(
    es, 
    ps, 
    r0s,
    save_this_figure = False
  ) -> None:
  """
  Heatmap for R0 of Omicron depending wrt. pre-existing immunity and immune evasion
  :param list es: immune evasion values for the vertical axis (resultion)
  :param list ps: pre-existing immunity values for the horizontal axis (resultion)
  :param list r0s: R0-contours to be highlighted
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """

  # compute data
  reproduction_numbers = []

  for e in es:
   
    reproduction_numbers.append([
        r0_omicron_from_contour_relation(
          e = e,
          p = p_sa
        )
        for p_sa in ps]
    )

  # setup the coloring scheme
  my_levels = np.arange(0, math.ceil(r0_delta * ratio_omicron_per_delta_south_africa) + 1, 1)
  magic_color_count = round(1.1 * len(my_levels))
  colors = plt.cm.winter(np.linspace(0, 1, magic_color_count + 1))

  # ensure proper fontsize
  plt.rcParams.update({'font.size': 10})

  fig, ax = plt.subplots(
      dpi = figures_dpi if save_this_figure else 200, 
      figsize = (4, 4)
    )

  ax.contourf(ps, es, reproduction_numbers, 
      levels = my_levels, 
      colors = colors, alpha = 1)
  contours = ax.contour(ps, es, reproduction_numbers,
      r0s, 
      colors='#e0e0e0', linewidths = 1.2)
  ax.clabel(contours, inline = True, fmt = str, fontsize = 7)

  ax.set_ylabel("immune evasion")
  ax.set_xlabel("pre-existing immunity in South Africa")

  ax.margins(0)

  plt.tight_layout()

  if save_this_figure == False:
    
    ax.set_title('$R_0$ of Omicron')

  else:
    
    myFileName = "contourRelationHeatmap.pdf"

    plt.savefig(myFileName, dpi=figures_dpi)
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# #### Figure

# In[ ]:


interact(
    lambda production : plot_r0_omicron_vs_immune_evasion(
        es = e_vals, 
        ps = [0.75, 0.8, 0.85, 0.9, 0.95],
        save_this_figure = production
    ),
    production = False
)


# In[ ]:


interact(
    lambda production : heatmap_r0_omicron_vs_immune_evasion(
        es = np.linspace(0, 1, 100),
        ps = np.linspace(0, 0.98, 100),
        r0s = [3, 6, 9, 12, 15, 18, 20, 22, 23, 23.5, 23.8],
        save_this_figure = production
      ),
      production = False           
)


# ### Level of non-pharmaceutical interventions (NPI) required to suppress Delta

# #### Code

# In[ ]:


def plot_omicron_suppressing_npi(
    ps,
    es,
    p_sa: float = p_south_africa,
    r0_delta = r0_delta,
    save_this_figure = False
  ) -> None:
  """
  Plot of Omicron suppressing suppressing NPIs compared to the NPI suppressing Delta
  :param list ps: pre-existing immunity values for the horizontal axis (resultion)
  :param list es: immune evasion values (number of curves)
  :param float p_sa: pre-existing immunity in South Africa
  :param float r0_delta: R0 of the Delta variant
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """

  # compute the npi suppressing Delta for all model locations (ps)
  npi_suppressing_delta = np.array([
      calculate_suppressing_npi(
          r0 = r0_delta, 
          p = p
      ) 
      for p in ps
  ])

  # ensure proper fontsize
  plt.rcParams.update({'font.size': 10})

  # setup the coloring scheme
  magic_color_count = round(1.5 * len(es))
  colors = plt.cm.winter(np.linspace(0, 1, magic_color_count + 1))

  # setup the figure
  plt.figure(
      dpi = figures_dpi if save_this_figure else 150, 
      figsize=(5, 3)
    )

  # plot a curve for each e \in es
  for idx, e in enumerate(es):
    
    # Get R0 of the Omicron variant
    r0_omicron = r0_omicron_from_contour_relation(
      p = p_sa,
      e = e
    )

    # compute the npi suppressing Delta for all model locations (ps)
    npi_suppressing_omicron = np.array([
        calculate_suppressing_npi(
            r0 = r0_omicron,
            p = p * (1 - e)
        )
        for p in ps
    ])
    
    plt.plot(ps, npi_suppressing_omicron, 
             label=str(round(e, 1)), 
             color=colors[magic_color_count - idx])

  # plot a curve for Delta suppression
  plt.plot(ps, npi_suppressing_delta, 'r--', 
           linewidth=3, 
           label="suppression of $\Delta$")

  lgd = plt.legend(loc='right', bbox_to_anchor=(1.55, 0.5),
                   title='Immune evasion\nof the Omicron variant')

  plt.xlim(ps[0], ps[-1])
  plt.ylim(0, 1)

  plt.xlabel('pre-existing immunity')
  plt.ylabel('reduction of transmission by NPIs')

  if save_this_figure == False:
    
    plt.title('NPI requirement for controlling Omicron')

  else:
    
    myFileName = "npiRequirementPlot.pdf"

    plt.savefig(myFileName, dpi=figures_dpi, 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# #### Figures

# In[ ]:


interact(
  lambda p_sa = p_south_africa, production = False : plot_omicron_suppressing_npi(
      ps = np.linspace(0.4, 1, 1000), 
      es = np.arange(0.2, 0.8, 0.1), 
      p_sa = p_sa,
      save_this_figure = production
  ),
  p_sa = (0, 1, 0.01),
  production = False
)


# ### Timeplots of the Omicron model

# #### Code

# In[ ]:


def plot_omicron_model_on_axes(
    ax, 
    p_loc, 
    e, 
    t, 
    y_range = 100,
    title_prefix = '',
    title_r0 = False
  ) -> None:
  """
  Timeplot of Omicron spread on given figure
  :param ax: axes of the figure
  :param float p_loc: pre-existing immunity of the model country
  :param list e: immune evasion ratio of Omicron
  :param list t: time range and resolution
  :param float y_range: sets the y-range of the main plot (I-plot)
  :param str title_prefix: prepends title
  :param bool title_r0: adds R_0, R_t of Omicron to title
  :return None
  """
  
  # local npi
  npi_loc = calculate_suppressing_npi(
      r0 = r0_delta, 
      p = p_loc
    )

  # r0 omicron
  r0_omicron = r0_omicron_from_contour_relation(e = e)

  # Get model solution
  sol = solve_omicron_model(
      r0_omicron = r0_omicron,
      e = e,
      p_loc = p_loc,
      npi_loc = npi_loc,
      initial_l1 = 0.00001,
      t = t
    )
  
  sol_d = {comps[i]: sol[:, i] for i in range(len(comps))}

  # get the timeseries for compartments
  s   = sol_d["s"]
  l_s = sol_d["l1_s"] + sol_d["l2_s"]
  i_s = sol_d["i1_s"] + sol_d["i2_s"] + sol_d["i3_s"] + sol_d["i4_s"]
  r_s = sol_d["r_s"]

  p   = sol_d["p"]
  l_p = sol_d["l1_p"] + sol_d["l2_p"]
  i_p = sol_d["i1_p"] + sol_d["i2_p"] + sol_d["i3_p"] + sol_d["i4_p"]
  r_p = sol_d["r_p"]
  
  # main plot
  color_map = ["#ff6666", "#ffaaaa"]

  plt.rcParams.update({'font.size': 7})

  ax.stackplot(t, i_s * 100., i_p * 100., colors = color_map)

  ax.set_xlabel("time (days)")
  ax.set_ylabel("infected (%)")

  title = title_prefix + "p=" + str(p_loc) + ", e=" + str(e) + ", npi=" + "{:.2f}".format(npi_loc)

  if title_r0:
    title = title + ", $R_0$=" + "{:.2f}".format(r0_omicron) +                     ", $R_{t^*}$=" + "{:.2f}".format(r0_omicron * (1 - npi_loc) * (1 - p_loc + e * p_loc))

  ax.set_title(title)

  ax.set_xlim([0, t[-1]])
  ax.set_ylim([0, y_range])

  # create the inset
  left, bottom, width, height = [0.55, 0.55, 0.40, 0.40]
  ax2 = ax.inset_axes([left, bottom, width, height])
  
  color_map_inset = color_map + ["#ffffff", "#dfdfdf", "#d0d0d0"]

  ax2.stackplot(t, 
                r_s * 100, 
                r_p * 100, 
                (s + l_s + i_s + l_p + i_p) * 100, 
                p * 100,
                np.full(r_s.shape, (1 - e) * p_loc * 100),
                colors = color_map_inset) 

  ax2.set_ylabel("affected (%)")

  ax2.set_xlim([0, t[-1]])
  ax2.set_ylim([0, 100])


# In[ ]:


def plot_omicron_model(
    p_loc = 0.5,
    e = 0.5,
    T = 200,
    y_range = 20,
    title_prefix = '',
    title_r0 = True,
    save_this_figure = False
  ) -> None:
  """
  Timeplot of Omicron spread
  :param float p_loc: pre-existing immunity of the model country
  :param list e: immune evasion ratio of Omicron
  :param float T: final simulation time
  :param float y_range: sets the y-range of the main plot (I-plot)
  :param str title_prefix: prepends title
  :param bool title_r0: adds R_0, R_t of Omicron to title
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """
  
  fig = plt.figure(
      dpi = figures_dpi if save_this_figure else 150, 
      figsize=(4, 4))
  
  ax = plt.gca()

  plt.rcParams.update({'font.size': 7})

  plot_omicron_model_on_axes(
      ax = ax,
      p_loc = p_loc, 
      e = e,
      t = np.linspace(0, T, 200),
      y_range = y_range,
      title_prefix = title_prefix,
      title_r0 = title_r0
    )
  
  if save_this_figure:
    
    myFileName = "singleTimeplot.pdf"

    plt.savefig(myFileName, dpi=figures_dpi)
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# In[ ]:


def multiplot_omicron_model(
    ps, 
    es,
    title_prefixes,
    T,
    y_range,
    title_r0 = False,
    save_this_figure = False
  ) -> None:
  """
  4 timeplots of Omicron spread (4 scenarios)
  :param list ps: pre-existing immunity levels of model countries (4-list)
  :param list e: immune evasion ratios of Omicron (4-list)
  :param list title_prefixes: prefixes to titles (4-list)
  :param float T: final simulation time (common)
  :param float y_range: sets the y-range of the main I-plots (common)
  :param bool title_r0: adds R_0, R_t of Omicron to titles (common)
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """

  fig = plt.figure(
      dpi = figures_dpi if save_this_figure else 110, 
      figsize = (7, 7)
    )
  
  plt.rcParams.update({'font.size': 7})

  if not isinstance(title_prefixes, list):
    title_prefixes = ['', '', '', '']

  t = np.linspace(0, T, 1000)

  ax = fig.add_subplot(221)
  plot_omicron_model_on_axes(ax = ax, p_loc = ps[0], e = es[0], t = t,
                             title_prefix = title_prefixes[0], 
                             y_range = y_range, title_r0 = title_r0)
  ax = fig.add_subplot(222)
  plot_omicron_model_on_axes(ax = ax, p_loc = ps[1], e = es[1], t = t, 
                             title_prefix = title_prefixes[1], 
                             y_range = y_range, title_r0 = title_r0)
  ax = fig.add_subplot(223)
  plot_omicron_model_on_axes(ax = ax, p_loc = ps[2], e = es[2], t = t,
                             title_prefix = title_prefixes[2],  
                             y_range = y_range, title_r0 = title_r0)
  ax = fig.add_subplot(224)
  plot_omicron_model_on_axes(ax = ax, p_loc = ps[3], e = es[3], t = t, 
                             title_prefix = title_prefixes[3], 
                             y_range = y_range, title_r0 = title_r0)
  
  fig.tight_layout()

  if save_this_figure:
    
    myFileName = "fourTimeplots.pdf"

    plt.savefig(myFileName, dpi=figures_dpi)
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# #### Figures

# In[ ]:


interact(
    plot_omicron_model, 
    p_loc = (0, 1, 0.01), 
    e = (0, 1, 0.01), 
    T = (0, 500, 1), 
    y_range = (0, 100, 1)
  )


# In[ ]:


interact(
    lambda production = False : multiplot_omicron_model(
        ps = [0.1, 0.75, 0.9, 0.96],
        es = [0.03, 0.08, 0.47, 0.68],
        title_prefixes = ['a) ', 'b) ', 'c) ', 'd) '],
        T = 75,
        y_range = 60,
        title_r0 = True,
        save_this_figure = production
    ),
    production = False
)


# In[ ]:


interact(
    lambda production = False : multiplot_omicron_model(
        ps = [0.6, 0.9, 0.6, 0.9],
        es = [0.8, 0.8, 0.5, 0.5],
        title_prefixes = ['a) ', 'b) ', 'c) ', 'd) '],
        T = 175,
        y_range = 60,
        title_r0 = False,
        save_this_figure = production
    ),
    production = False
)


# ### Analysis of peak and final size

# #### Code

# ##### Data generators

# In[ ]:


def calculate_for_fixed_e_all_peak_and_final_sizes(
    e, 
    ps, 
    p_sa = p_south_africa,
    severity: float = 1, 
    relative_severity: float = (1 - omicron_hospital_evasion)
  ) -> tuple:
  """
  Calculates peak and final sizes for fixed immune evasion
  :param float e: immune evasion of Omicron
  :param list ps: pre-existing immunity values for the horiztonal axis
  :param float p_sa: pre-existing immunity in South Africa
  :param float severity: common weight of _s and _p compartments
  :param float relative_severity: additional weight of _p compartments
  :return tuple: list of peak sizes and list of final sizes
  """
  # R_0 of Omicron
  r0_omicron = r0_omicron_from_contour_relation(
      e = e,
      p = p_sa
    )

  peak_sizes = []
  final_sizes = []

  for p_loc in ps:

    # local npi
    npi_loc = calculate_suppressing_npi(
        r0 = r0_delta, 
        p = p_loc
      )

    # numerical solution
    sol = solve_omicron_model(
        r0_omicron = r0_omicron,
        e = e,
        p_loc = p_loc,
        npi_loc = npi_loc,
        initial_l1 = 0.00001,
        t = t
      )
    
    peak_size, final_size = calculate_peak_and_final_size(
      sol = sol,
      severity = severity,
      relative_severity = relative_severity)
    
    peak_sizes.append(peak_size)
    final_sizes.append(final_size)

  return peak_sizes, final_sizes


# In[ ]:


def generate_heatmap_data(
    severity: float = 1, 
    relative_severity: float = (1 - omicron_hospital_evasion)
  ) -> tuple:
  """
  Generates data for heatmaps for the chart [p_loc_vals, e_vals]
  :param float severity: common weight of _s and _p compartments
  :param float relative_severity: additional weight of _p compartments
  :return tuple: list of peak sizes, list of final sizes, list of reproduction numbers
  """
  peak_sizes = []
  final_sizes = []
  reproduction_numbers = []

  for e in e_vals:

    # peak and final size
    peaks, finals = calculate_for_fixed_e_all_peak_and_final_sizes(
        e = e,
        ps = p_loc_vals, 
        severity = severity, 
        relative_severity = relative_severity)
    
    peak_sizes.append(peaks)
    final_sizes.append(finals)

    # R_0 of Omicron
    r0_omicron = r0_omicron_from_contour_relation(
      e = e,
      p = p_south_africa
    )
   
    # R_{t^*} in model countries
    reproduction_numbers.append(
        [r0_omicron * (1 - calculate_suppressing_npi(
            r0 = r0_delta,
            p = p_loc
        )) * (1 - p_loc + e * p_loc)
        for p_loc in p_loc_vals]
    )

  return np.array(peak_sizes), np.array(final_sizes), np.array(reproduction_numbers)


# ##### Figure generators

# In[ ]:


def plot_peak_and_final_size(
    e: float, 
    p_sa : float = p_south_africa,
    severity: float = 1, 
    relative_severity: float = (1 - omicron_hospital_evasion),
    y_limit_peak = 1.,
    y_limit_final = 1.,
    save_this_figure = False
  ) -> None:
  """
  Plot of peak and final size wrt. pre-existing immunity in model country (p_loc)
  :param float p_sa: pre-existing immunity in South Africa
  :param float severity: common weight of _s and _p compartments
  :param float relative_severity: additional weight of _p compartments
  :param list y_limit_peak: ymax for the peak size
  :param list y_limit_final: ymax for the final size
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """
  peak_sizes, final_sizes = calculate_for_fixed_e_all_peak_and_final_sizes(
      e = e, 
      ps = p_loc_vals, 
      p_sa = p_sa, 
      severity = severity, 
      relative_severity = relative_severity
  )

  fig = plt.figure(
      dpi = figures_dpi if save_this_figure else 110, 
      figsize = (5, 3))

  plt.rcParams.update({'font.size': 7})

  ax = fig.add_subplot(121)
  
  # peak sizes
  ax.plot(p_loc_vals, peak_sizes)

  ax.set_xlabel("pre-existing immunity")
  ax.set_title("peak size")

  ax.set_ylim(0.0, y_limit_peak)

  ax = fig.add_subplot(122)
  # final sizes
  ax.plot(p_loc_vals, final_sizes)

  ax.set_xlabel("pre-existing immunity")
  ax.set_title("final size")

  ax.set_ylim(0.0, y_limit_final)

  # finalize
  fig.tight_layout()

  if save_this_figure:
    
    myFileName = "peakAndFinalSize.pdf"

    plt.savefig(myFileName, dpi=figures_dpi)
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# In[ ]:


def plot_heatmap(
    data: np.ndarray,
    type = "final",
    add_frame: dict = None,
    add_npi_plot: bool = True,
    save_this_figure = False
  ) -> None:
  """
  Generate heatmap of given type from the data
  :param np.ndarray data: data given as [[data(p, e) for p_loc_vals] for e_vals]
  :param str type: final, peak, reproduction_number
  :param dict add_frame: None or dictionary describing a highlighted frame
  :param bool add_npi_plot: adding a plot of Delta suppressing npis
  :param bool save_this_figure: if True then the figure is saved
  :return None
  """
  
  this_figure_dpi = figures_dpi if save_this_figure else 100

  if add_npi_plot:
    fig, (ax1, ax) = plt.subplots(2, sharex = True, dpi = this_figure_dpi,
                                 figsize = (5, 7.5), 
                                 gridspec_kw = {'height_ratios': [1, 2]})
    
    # NPI plot
    ax1.plot(p_loc_vals, 
            [calculate_suppressing_npi(
                r0 = r0_delta, 
                p = p_loc) for p_loc in p_loc_vals])
    
    ax1.set_ylabel("NPI controlling Delta")

    ax1.set_ylim(0, 1.)

    ax1.margins(0)

  else:
    fig, ax = plt.subplots(dpi = this_figure_dpi,
                           figsize = (7, 7))


  plt.rcParams.update({'font.size': 11})

  # final size
  if type == "final": 
    frame_color = "white"
    marker_color = "white"
    
    levels = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    colormap = 'Reds'

    curves = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    curve_color = '#4a4a4a'

    title = 'final size'

  # peak size
  elif type == "peak":
    frame_color = "#6e6e6e"
    marker_color = "#6e6e6e"
    
    levels = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] 
    colormap = 'Oranges'

    curves = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] 
    curve_color = '#5e5e5e'

    title = 'peak size'

  # reproduction number
  else:
    frame_color = "black"
    marker_color = "black"

    levels = [1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
    colormap = 'Purples'

    curves = [1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
    curve_color = '#5e5e5e'

    title = 'control reproduction number'

  # generate the main heatmap
  ax.contourf(p_loc_vals, e_vals, data, 
      levels = levels, 
      cmap = colormap, alpha = 1)
  
  contours = ax.contour(p_loc_vals, e_vals, data,
      curves, 
      colors = curve_color, linewidths = 1)
  
  ax.clabel(contours, inline = True, fmt = str, fontsize = 8)
  
  ax.set_ylabel("immune evasion")
  ax.set_xlabel("pre-existing immunity")
  
  if not save_this_figure:
    ax.set_title(title, fontsize = 15)

  ax.margins(0)

  # label axes with %
  positions = [0, 0.25, 0.5, 0.75, 1]
  labels = ["0%", "25%", "50%", "75%", "100%"]

  ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
  ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))

  ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
  ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

  # add highlighting frame
  if add_frame is not None:
    frame_p = add_frame["frame_p"]
    frame_e = add_frame["frame_e"]

    markers = add_frame["markers"]

    highlighted_area = Rectangle(
      (frame_p[0], frame_e[0]),
      frame_p[1] - frame_p[0], frame_e[1] - frame_e[0],
      fc = 'none', 
      ec = frame_color,
      lw = 5,
      alpha = 0.5)
    
    ax.add_patch(highlighted_area)

    for marker in markers:
      ax.text(marker["p"] + 0.01, marker["e"]+ 0.01, s = marker["name"], fontsize = 12, color= marker_color)
      ax.plot(marker["p"], marker["e"], "x", color = marker_color)
      
  # finalize
  fig.tight_layout()

  if save_this_figure:
    
    myFileName = "heatmap-" + type + ".pdf"

    plt.savefig(myFileName, dpi = figures_dpi)
    
    if figures_autodownload and use_colab:
      files.download(myFileName)


# #### Figures

# ##### Plot of peak and final sizes for fixed immune evasion

# In[ ]:


interact(
    plot_peak_and_final_size,
    e = (0.2, 1, 0.01), 
    p_sa = (0, 1, 0.01),
    severity = (0, 1, 0.01), 
    relative_severity = (0, 1, 0.01),
    y_limit_peak = (0, 1, 0.01),
    y_limit_final = (0, 1, 0.01)
)


# ##### Heatmaps for peak size, final size, and control reproduction number of Omicron

# ###### Data generation [slow ~ 2 x 1m 30s]

# In[ ]:


# generate data considering the population not immune to Omicron
peak_sizes, final_sizes, reproduction_numbers = generate_heatmap_data(severity = 1, relative_severity = 1)


# In[ ]:


# generate data considering the population not immune to Delta
peak_sizes_s_only, final_sizes_s_only, reproduction_numbers = generate_heatmap_data(severity = 1, relative_severity = 0)


# ###### Heatmaps

# In[ ]:


frame_to_add = { 
    "frame_p": [0.3, 0.95],   
    "frame_e": [0.4, 0.95],
    "markers": [
                 {"p": 0.6, "e": 0.8, "name": "a"},
                 {"p": 0.9, "e": 0.8, "name": "b"},
                 {"p": 0.6, "e": 0.5, "name": "c"},
                 {"p": 0.9, "e": 0.5, "name": "d"}
               ]
}

print('CONTROL REPRODUCTION NUMBER')

interact(
    lambda add_npi_plot, add_frame, production : plot_heatmap(
        data = reproduction_numbers, 
        type = "reproduction_number",
        add_frame = (frame_to_add if add_frame else None),
        add_npi_plot = add_npi_plot,
        save_this_figure = production
      ),
      add_npi_plot = True,
      add_frame = True,
      production = False
)


# In[ ]:


frame_to_add = { 
    "frame_p": [0.3, 0.95],   
    "frame_e": [0.4, 0.95],
    "markers": [
                 {"p": 0.6, "e": 0.8, "name": "a"},
                 {"p": 0.9, "e": 0.8, "name": "b"},
                 {"p": 0.6, "e": 0.5, "name": "c"},
                 {"p": 0.9, "e": 0.5, "name": "d"}
               ]
}

print('PEAK SIZE FOR SEVERITY = 1, RELATIVE SEVERITY = 1')

interact(
    lambda add_npi_plot, add_frame, production : plot_heatmap(
        data = peak_sizes, 
        type = "peak",
        add_frame = (frame_to_add if add_frame else None),
        add_npi_plot = add_npi_plot,
        save_this_figure = production
      ),
      add_npi_plot = True,
      add_frame = True,
      production = False
)


# In[ ]:


frame_to_add = { 
    "frame_p": [0.3, 0.95],   
    "frame_e": [0.4, 0.95],
    "markers": [
                 {"p": 0.6, "e": 0.8, "name": "a"},
                 {"p": 0.9, "e": 0.8, "name": "b"},
                 {"p": 0.6, "e": 0.5, "name": "c"},
                 {"p": 0.9, "e": 0.5, "name": "d"}
               ]
}

print('FINAL SIZE FOR SEVERITY = 1, RELATIVE SEVERITY = 1')

interact(
    lambda add_npi_plot, add_frame, production : plot_heatmap(
        data = final_sizes, 
        type = "final",
        add_frame = (frame_to_add if add_frame else None),
        add_npi_plot = add_npi_plot,
        save_this_figure = production
      ),
      add_npi_plot = True,
      add_frame = True,
      production = False
)


# In[ ]:


frame_to_add = { 
    "frame_p": [0.3, 0.95],   
    "frame_e": [0.4, 0.95],
    "markers": [
                 {"p": 0.6, "e": 0.8, "name": "a"},
                 {"p": 0.9, "e": 0.8, "name": "b"},
                 {"p": 0.6, "e": 0.5, "name": "c"},
                 {"p": 0.9, "e": 0.5, "name": "d"}
               ]
}

print('PEAK SIZE FOR SEVERITY = 1, RELATIVE SEVERITY = 0')

interact(
    lambda add_npi_plot, add_frame, production : plot_heatmap(
        data = peak_sizes_s_only, 
        type = "peak",
        add_frame = (frame_to_add if add_frame else None),
        add_npi_plot = add_npi_plot,
        save_this_figure = production
      ),
      add_npi_plot = True,
      add_frame = True,
      production = False
)


# In[ ]:


frame_to_add = { 
    "frame_p": [0.3, 0.95],   
    "frame_e": [0.4, 0.95],
    "markers": [
                 {"p": 0.6, "e": 0.8, "name": "a"},
                 {"p": 0.9, "e": 0.8, "name": "b"},
                 {"p": 0.6, "e": 0.5, "name": "c"},
                 {"p": 0.9, "e": 0.5, "name": "d"}
               ]
}

print('FINAL SIZE FOR SEVERITY = 1, RELATIVE SEVERITY = 0')

interact(
    lambda add_npi_plot, add_frame, production : plot_heatmap(
        data = final_sizes_s_only, 
        type = "final",
        add_frame = (frame_to_add if add_frame else None),
        add_npi_plot = add_npi_plot,
        save_this_figure = production
      ),
      add_npi_plot = True,
      add_frame = True,
      production = False
)

