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

# In[ ]:


from typing import Union

from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from scipy.integrate import odeint


# ## Parametrization

# ### Epidemiological Parameters

# In[ ]:


# Delta variant

# basic reproduction number of the Delta variant 
#  (relevant for fully susceptible population with no interventions in place)
r0_delta_glob = 6.0


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
alpha_glob = 1. / omicron_latent_period

# gamma
gamma_glob = 1. / omicron_infectious_period


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
t_glob = np.linspace(0, 500, 5000)

# Model compartments
comps = ["s", "l1_s", "l2_s", "i1_s", "i2_s", "i3_s", "i4_s", "r_s",
         "p", "l1_p", "l2_p", "i1_p", "i2_p", "i3_p", "i4_p", "r_p"]


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
        e: Union[float, np.ndarray],
        p: float = p_south_africa,
        r0_delta: float = r0_delta_glob,
        ratio_omicron_per_delta: float = ratio_omicron_per_delta_south_africa
    ) -> float:
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


def calculate_suppressing_npi(
        r0: float,
        p: float,
        goal: float = 1
    ) -> float:
    """
    Calculate the necessary contact rate reduction to achieve the <goal> rep. number
    :param float r0: basic reproduction number
    :param float p: pre-existing immunity
    :param float goal: desired reproduction number (<= 1)
    :return float: NPI
    """
    return 0 if (p == 1) else 1 - np.min((1.0, goal / (r0 * (1 - p))))


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
    s, l1_s, l2_s, i1_s, i2_s, i3_s, i4_s, r_s,     p, l1_p, l2_p, i1_p, i2_p, i3_p, i4_p, r_p = xs

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

    return np.array([ds, dl1_s, dl2_s, di1_s, di2_s, di3_s, di4_s, dr_s,
                   dp, dl1_p, dl2_p, di1_p, di2_p, di3_p, di4_p, dr_p])


# In[ ]:


def calculate_beta(
        r0: float,
        params: dict
    ) -> float:
    """
    Calculate beta from R0 and other parameters
    :param float r0: basic reproduction number
    :param dict params: dictionary of parameters
    :return float: calculated beta
    """

    return r0 * params["gamma"]


# In[ ]:


def solve_omicron_model(
        r0_omicron: float,
        e: Union[float, np.ndarray],
        p_loc: float,
        npi_loc: float,
        initial_l1: float,
        t: np.ndarray = t_glob
    ) -> list:
    """
    Calculate peak and final sizes
    :param float r0_omicron:
    :param float e:
    :param float p_loc:
    :param float npi_loc:
    :param float initial_l1:
    :param np.ndarray t:
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

    iv = [s_0, l1_s_0, l2_s_0, i1_s_0, i2_s_0, i3_s_0, i4_s_0, r_s_0,
        p_0, l1_p_0, l2_p_0, i1_p_0, i2_p_0, i3_p_0, i4_p_0, r_p_0]

    # set readily known parameters
    params = {
      "alpha": alpha_glob,
      "gamma": gamma_glob,
      "npi": npi_loc
    }

    # calculate beta
    beta = calculate_beta(
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
        sol: np.ndarray,
        severity: float = 1,
        relative_severity: float = (1 - omicron_hospital_evasion)
    ) -> list:
    """
    Calculate peak and final sizes
    :param np.ndarray sol: solution of the numerical simulation
    :param float severity: common weight of _s and _p compartments
    :param float relative_severity: additional weight of _p compartments
    :return list: peak and final size
    """

    # unwrap the ODE solution
    sol_d = {comps[i]: sol[:, i] for i in range(len(comps))}

    # plug-in weights
    r = severity * (sol_d["r_s"] + relative_severity * sol_d["r_p"])

    i = severity * (
          sol_d["i1_s"] + sol_d["i2_s"] + sol_d["i3_s"] + sol_d["i4_s"] +
          relative_severity * (sol_d["i1_p"] + sol_d["i2_p"] + sol_d["i3_p"] + sol_d["i4_p"])
    )

    # peak size
    peak_size = np.max(i)

    # final size
    final_size = r[-1]

    return [peak_size, final_size]


# ## Results

# ### Contours: R0 of Omicron vs immune evasion

# #### Code

# In[ ]:


def plot_r0_omicron_vs_immune_evasion(
        es: np.ndarray,
        ps: Union[np.ndarray, list],
        save_this_figure = False
    ) -> None:
    """
    Plot R0 of Omicron depending on its immune evasion
    :param np.ndarray es: immune evasion values for the horizontal axis (resultion)
    :param np.ndarray ps: pre-existing immunity values (number of curves)
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
            r0_delta = r0_delta_glob,
            ratio_omicron_per_delta = ratio_omicron_per_delta_south_africa
        )

        ax.plot(es, r0_omicron_vals,
                 label=str(round(p, 2)),
                 color=colors[magic_color_count - idx])

        lgd = ax.legend(loc='right', bbox_to_anchor=(1.6, 0.5),
                  title='Pre-existing immunity\nin South Africa\n(fraction of population)')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, r0_delta_glob * ratio_omicron_per_delta_south_africa)

        ax.set_yticks(range(0, int(r0_delta_glob * ratio_omicron_per_delta_south_africa) + 1, 4))

        ax.set_xlabel('immune evasion')
        ax.set_ylabel('$R_0$ of Omicron')

        if not save_this_figure:

            ax.set_title('Immune Evasion vs $R_0$ of Omicron')
        else:
            my_file_name = "contourRelation.pdf"

            plt.savefig(my_file_name, dpi=figures_dpi,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')

            # if figures_autodownload:
            #     from google.colab import files
            #     files.download(my_file_name)


# In[ ]:


def heatmap_r0_omicron_vs_immune_evasion(
        es: np.ndarray,
        ps: Union[np.ndarray, list],
        r0s: list,
        save_this_figure: bool = False
    ) -> None:
    """
    Heatmap for R0 of Omicron depending wrt. pre-existing immunity and immune evasion
    :param np.ndarray es: immune evasion values for the vertical axis (resultion)
    :param np.ndarray ps: pre-existing immunity values for the horizontal axis (resultion)
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
    my_levels = np.arange(0, np.ceil(r0_delta_glob * ratio_omicron_per_delta_south_africa) + 1, 1)
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

    if not save_this_figure:
        ax.set_title('$R_0$ of Omicron')
    else:
        my_file_name = "contourRelationHeatmap.pdf"
        plt.savefig(my_file_name, dpi=figures_dpi)

        # if figures_autodownload:
        #   files.download(myFileName)


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
        ps: Union[np.ndarray, list],
        es: Union[np.ndarray, list],
        p_sa: float = p_south_africa,
        r0_delta = r0_delta_glob,
        save_this_figure = True
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

    if not save_this_figure:
        plt.title('NPI requirement for controlling Omicron')
    else:
        my_file_name = "npiRequirementPlot.pdf"
        plt.savefig(my_file_name, dpi=figures_dpi,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')

        # if figures_autodownload:
        #     files.download(myFileName)


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
        p_loc: float,
        e: Union[list, np.ndarray],
        t: np.ndarray,
        y_range: int = 100,
        title_prefix: str = '',
        add_r0_omicron_to_title: bool = False
    ) -> None:
    """
    Plot omicron model on input axes
    :param plt.axes.Axes ax:
    :param float p_loc:
    :param Union[list, np.ndarray] e:
    :param np.ndarray t:
    :param int y_range:
    :param str title_prefix:
    :param bool add_r0_omicron_to_title:
    :return: None
    """
  
    # local npi
    npi_loc = calculate_suppressing_npi(
      r0 = r0_delta_glob,
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

    ax.stackplot(t, i_s * 100., i_p * 100., colors = color_map)

    ax.set_xlabel("time (days)")
    ax.set_ylabel("infected (%)")

    title = title_prefix + "p=" + str(p_loc) + ", e=" + str(e) + ", npi=" + "{:.2f}".format(npi_loc)

    if add_r0_omicron_to_title:
        title = title + ", $R_0$=" + "{:.2f}".format(r0_omicron) +                         ", $R_{t^*}$=" + "{:.2f}".format(r0_omicron * (1 - npi_loc) * (1 - p_loc + e * p_loc))

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
        p_loc: Union[int, float] = 0.5,
        e: Union[float, np.ndarray] = 0.5,
        T: Union[int, float] = 200,
        y_range: Union[int, float] = 20,
        title_prefix: str = '',
        add_r0_omicron_to_title: bool = True,
        save_this_figure: bool = False
    ) -> None:
    """
    Plot omicron model
    :param Union[int, float] p_loc:
    :param Union[float, np.ndarray] e:
    :param Union[int, float] T:
    :param Union[int, float] y_range:
    :param str title_prefix:
    :param bool add_r0_omicron_to_title:
    :param bool save_this_figure:
    :return:
    """
    fig = plt.figure(
      dpi=figures_dpi if save_this_figure else 150,
      figsize=(3, 3))

    ax = plt.gca()

    plt.rcParams.update({'font.size': 5})

    plot_omicron_model_on_axes(
      ax = ax,
      p_loc = p_loc,
      e = e,
      t = np.linspace(0, T, 200),
      y_range = y_range,
      title_prefix = title_prefix,
      add_r0_omicron_to_title = add_r0_omicron_to_title
    )
  
    if not save_this_figure:
        plt.rcParams.update({'font.size': 5})
    else:
        my_file_name = "singleTimeplot.pdf"
        plt.savefig(my_file_name, dpi=figures_dpi)
    
    # if figures_autodownload:
    #   files.download(myFileName)
    plt.show()


# In[ ]:


def multiplot_omicron_model(
        ps: Union[np.ndarray, list],
        es: Union[np.ndarray, list],
        y_range: int,
        add_r0: bool,
        t: np.ndarray,
        save_this_figure: bool = False
    ) -> None:
    """
    Multiple plot
    :param Union[np.ndarray, list] ps:
    :param Union[np.ndarray, list] es:
    :param int y_range:
    :param bool add_r0:
    :param np.ndarray t:
    :param bool save_this_figure:
    :return: None
    """
    fig = plt.figure(
      dpi=figures_dpi if save_this_figure else 110,
      figsize=(7, 7)
    )
  
    plt.rcParams.update({'font.size': 7})

    ax = fig.add_subplot(221)
    plot_omicron_model_on_axes(ax=ax, p_loc=ps[0], e=es[0], t=t,
                             y_range=y_range, add_r0_omicron_to_title=add_r0)
    ax = fig.add_subplot(222)
    plot_omicron_model_on_axes(ax=ax, p_loc=ps[1], e=es[1], t=t,
                             y_range=y_range, add_r0_omicron_to_title=add_r0)
    ax = fig.add_subplot(223)
    plot_omicron_model_on_axes(ax=ax, p_loc=ps[2], e=es[2], t=t,
                             y_range=y_range, add_r0_omicron_to_title=add_r0)
    ax = fig.add_subplot(224)
    plot_omicron_model_on_axes(ax=ax, p_loc=ps[3], e=es[3], t=t,
                             y_range=y_range, add_r0_omicron_to_title=add_r0)

    fig.tight_layout()

    if save_this_figure:
        my_file_name = "fourTimeplots.pdf"
        plt.savefig(my_file_name, dpi=figures_dpi)
    
    # if figures_autodownload:
    #     files.download(myFileName)


# #### Figures

# In[ ]:


# TODO: does not work
# interact(
#     plot_omicron_model, 
#     p_loc = (0, 1, 0.01), 
#     e = (0, 1, 0.01), 
#     T = (0, 500, 1), 
#     y_range = (0, 100, 1)
#   )


# In[ ]:


interact(
    lambda production = False : multiplot_omicron_model(
        ps = [0.1, 0.75, 0.9, 0.96],
        es = [0.03, 0.08, 0.47, 0.68],
        t = np.linspace(0, 75, 1000),
        y_range = 60,
        add_r0 = True,
        save_this_figure = production
    ),
    production = False
)


# In[ ]:


interact(
    lambda production = False : multiplot_omicron_model(
        ps = [0.6, 0.9, 0.6, 0.9],
        es = [0.8, 0.8, 0.5, 0.5],
        t = np.linspace(0, 175, 1000),
        y_range = 60,
        add_r0 = True,
        save_this_figure = production
    ),
    production = False
)


# ### Analysis of peak and final size

# #### Code

# ##### Data generators

# In[ ]:


def calculate_for_fixed_e_all_peak_and_final_sizes(
        e: Union[list, np.ndarray, float],
        ps: Union[list, np.ndarray],
        p_sa: float = p_south_africa,
        severity: float = 1,
        relative_severity: float = (1 - omicron_hospital_evasion)
    ) -> list:
    """
    Interactive plot for relationship between peak and final size
    :param Union[list, np.ndarray, float] ps:
    :param Union[list, np.ndarray] e:
    :param float p_sa:
    :param float severity:
    :param float relative_severity:
    :return None
    """
    # r0 omicron
    r0_omicron = r0_omicron_from_contour_relation(
      e = e,
      p = p_sa
    )

    peak_sizes = []
    final_sizes = []

    for p_loc in ps:
        # local npi
        npi_loc = calculate_suppressing_npi(
            r0 = r0_delta_glob,
            p = p_loc
          )

        # Get model solution
        sol = solve_omicron_model(
            r0_omicron = r0_omicron,
            e = e,
            p_loc = p_loc,
            npi_loc = npi_loc,
            initial_l1 = 0.00001,
            t = t_glob
          )
    
        peak_size, final_size = calculate_peak_and_final_size(
          sol=sol,
          severity=severity,
          relative_severity=relative_severity)

        peak_sizes.append(peak_size)
        final_sizes.append(final_size)

    return [peak_sizes, final_sizes]


# In[ ]:


def generate_heatmap_data(
        severity: float = 1,
        relative_severity: float = (1-omicron_hospital_evasion),
        frame_p: list = None,
        frame_e: list = None
    ) -> tuple:
    """
    Generates data for heatmaps
    :param float severity:
    :param float relative_severity:
    :param list frame_p:
    :param list frame_e:
    :return tuple: tuple containing final sizes, peak sizes and reproduction numbers
    """
    if frame_e is None:
        frame_e = [0, 0]
    if frame_p is None:
        frame_p = [0, 0]
    peak_sizes = []
    final_sizes = []
    reproduction_numbers = []

    for e in e_vals:

        peaks, finals = calculate_for_fixed_e_all_peak_and_final_sizes(
            e=e,
            ps=p_loc_vals,
            severity=severity,
            relative_severity=relative_severity)

        peak_sizes.append(peaks)
        final_sizes.append(finals)

        # r0 omicron
        r0_omicron = r0_omicron_from_contour_relation(
          e = e,
          p = p_south_africa
        )

        reproduction_numbers.append(
            [r0_omicron * (1 - calculate_suppressing_npi(
                r0 = r0_delta_glob,
                p = p_loc
            )) * (1 - p_loc + e * p_loc)
            for p_loc in p_loc_vals]
        )

    return np.array(peak_sizes), np.array(final_sizes), np.array(reproduction_numbers)


# ##### Figure generators

# In[ ]:


def plot_peak_and_final_size(
        e: float,
        p_sa: float = p_south_africa,
        severity: float = 1,
        relative_severity: float = (1 - omicron_hospital_evasion),
        save_this_figure: bool = False
    ) -> None:
    """
    Interactive plot for relationship between peak and final size
    :param bool save_this_figure:
    :param float p_sa:
    :param float e:
    :param float severity:
    :param float relative_severity:
    :return None
    """
    peak_sizes, final_sizes =         calculate_for_fixed_e_all_peak_and_final_sizes(
            e=e, ps=p_loc_vals, p_sa = p_sa,
            severity=severity,
            relative_severity=relative_severity
    )

    fig = plt.figure(
      dpi=figures_dpi if save_this_figure else 110,
      figsize=(5, 3)
    )

    ax = fig.add_subplot(121)

    # Plot peak sizes
    ax.plot(p_loc_vals, peak_sizes)
    ax.set_xlabel("pre-existing immunity")
    ax.set_title("peak size")
    ax.set_ylim(0.0, 1.0)

    ax = fig.add_subplot(122)
    # Plot final sizes
    ax.plot(p_loc_vals, final_sizes)
    ax.set_xlabel("pre-existing immunity")
    ax.set_title("final size")
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()

    if save_this_figure:
        my_file_name = "peakAndFinalSize.pdf"
        plt.savefig(my_file_name, dpi=figures_dpi)

        # files.download(myFileName)


# In[ ]:


def plot_heatmap(
        typ: str,
        data: np.ndarray,
        frame_p: list = None,
        frame_e: list = None
    ) -> None:
    """
    Generates heatmaps
    :param str typ:
    :param np.ndarray data:
    :param list frame_p:
    :param list frame_e:
    :return None
    """
    if frame_e is None:
        frame_e = [0, 0]
    if frame_p is None:
        frame_p = [0, 0]
    positions = [0, 0.25, 0.5, 0.75, 0.99]
    labels = ["0%", "25%", "50%", "75%", "100%"]
    rectum = Rectangle(
    (frame_p[0], frame_e[0]),
    frame_p[1]-frame_p[0], frame_e[1]-frame_e[0],
    fc ='none',
    ec ='#ffffff',
    lw = 10)

    fig, (ax1, ax2) = plt.subplots(
        2,
        sharex=True, dpi=100,
        figsize=(5,7.5),
        gridspec_kw={'height_ratios': [1,2]})

    ax1.plot(p_loc_vals,
           [calculate_suppressing_npi(r0=r0_delta_glob, p=p_loc) for p_loc in p_loc_vals])
    ax1.set_title("Amount of non pharmaceutical interventions", fontsize=10)
    ax1.set_xlabel("Preexisting immunity of total population")
    ax1.set_ylabel("Amount of NPI")

    if typ == "final":
        c="white"
        ax2.contourf(p_loc_vals, e_vals, data,
            levels=[0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            cmap='Reds', alpha=1)
        contours2 = ax2.contour(p_loc_vals, e_vals, data,
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                colors='#4a4a4a', linewidths=1)
        ax2.clabel(contours2, inline=True, fmt=str, fontsize=8)
        ax2.set_title("Final size heatmap",
                      fontsize=10)
    elif typ == "peak":
        c="#6e6e6e"
        ax2.contourf(p_loc_vals, e_vals, data,
            levels=[0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            cmap='Oranges', alpha=1)
        contours = ax2.contour(p_loc_vals, e_vals, data,
            [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            colors='#5e5e5e', linewidths=1)
        ax2.clabel(contours, inline=True, fmt=str, fontsize=8)
        ax2.set_title("Peak size heatmap",
                        fontsize=10)
    else:
        c="black"
        ax2.contourf(p_loc_vals, e_vals, data,
            levels=[1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
            cmap='Purples', alpha=1)
        contours = ax2.contour(p_loc_vals, e_vals, data,
            [1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
            colors='#5e5e5e', linewidths=1)
        ax2.clabel(contours, inline=True, fmt=str, fontsize=8)
        ax2.set_title("Effective reproduction number heatmap",
                      fontsize=10)

    ax2.set_ylabel("Immune evasion")
    ax2.set_xlabel("Preexisting immunity of total population")

    ax2.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax2.yaxis.set_major_formatter(ticker.FixedFormatter(labels))

    ax2.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax2.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

    ax2.text(0.6, 0.7, s="a", fontsize=15, color= c)
    ax2.plot(0.6, 0.7, "x", color= c)
    ax2.text(0.85, 0.7, s="b", fontsize=15, color= c)
    ax2.plot(0.85, 0.7, "x", color= c)
    ax2.text(0.6, 0.3, s="c", fontsize=15, color= c)
    ax2.plot(0.6, 0.3, "x", color= c)
    ax2.text(0.85, 0.3, s="d", fontsize=15, color= c)
    ax2.plot(0.85, 0.3, "x", color= c)

    ax2.add_patch(rectum)

    ax1.margins(0)
    ax2.margins(0)

    plt.tight_layout()

    plt.show()
    plt.clf()


# #### Data generation [slow, expect 1m 30s]

# In[ ]:


peak_sizes_to_plot, final_sizes_to_plot, reproduction_numbers_to_plot =     generate_heatmap_data(severity = 1, relative_severity=1)


# #### Figure

# In[ ]:


# TODO: does not work
# interact(
#     plot_peak_and_final_size,
#     e=(0.2, 1, 0.01), 
#     p_sa=(0, 1, 0.01),
#     severity=(0, 1, 0.01), 
#     relative_severity=(0, 1, 0.01)
# )


# In[ ]:


plot_heatmap(data = peak_sizes_to_plot, typ="peak")


# In[ ]:


plot_heatmap(data = final_sizes_to_plot, typ="final")


# In[ ]:


plot_heatmap(data = reproduction_numbers_to_plot, typ="reproduction_number")


# In[ ]:


peak_sizes_s_only_to_plot, final_sizes_s_only_to_plot, reproduction_numbers_to_plot =     generate_heatmap_data(severity = 1, relative_severity=0)


# In[ ]:


plot_heatmap(data = peak_sizes_s_only_to_plot, typ="peak")


# In[ ]:


plot_heatmap(data = final_sizes_s_only_to_plot, typ="final")


# In[ ]:




