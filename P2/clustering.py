# -*- coding: utf-8 -*-
"""
Autor:
    Paula Villanueva Nuñez
    Jorge Casillas
Fecha:
    Noviembre/2021
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import cluster
import plotly.express as px
import warnings 

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
from math import floor
import seaborn as sns
from scipy.cluster import hierarchy
from scipy import stats


# Seleccionar casos
#caso = "caso1"
#caso = "caso2"
caso = "caso3"


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

def heatmap (X, alg, centers):
    print("---------- Heatmap de centroides...")
    
    centers_desnormal = centers.copy()

    # se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    #plt.figure()
    centers.index += 1
    #plt.figure()
    hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
    hm.set_ylim(len(centers),0)
    hm.figure.set_size_inches(15,15)
    hm.figure.savefig("img/" + caso + "/" + alg + "/centroides.pdf")
    plt.clf() 
    centers.index -= 1
    

def scatter_matrix (X_alg, alg, k):
    print("---------- Scatter matrix...")
    
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)
    #plt.figure()
    sns.set()
    variables = list(X_alg)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette=colors, plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.fig.set_size_inches(15,15)
    #plt.legend(labels=["Legend_Day1","Legend_Day2"], title = "Title_Legend")
    sns_plot.savefig("img/" + caso + "/" + alg + "/scatter.pdf")
    # plt.show()
    plt.clf()

def boxplot(X_alg, usadas, centers, alg, k, n_var):
    print("---------- Boxplots...")
    sns.set()
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)

    fig, axes = plt.subplots(k, n_var, sharey=True,figsize=(15,15))
    fig.subplots_adjust(wspace=0,hspace=0)

    #centers_sort = centers.sort_values(by=[usadas[0]]) #ordenamos para el plot

    rango = []
    for j in range(n_var):
        rango.append([X_alg[usadas[j]].min(),X_alg[usadas[j]].max()])

    for i in range(k):
        #c = centers_sort.index[i] 
        dat_filt = X_alg.loc[X_alg['cluster']==i]
        for j in range(n_var):
            #ax = sns.kdeplot(x=dat_filt[usadas[j]], label="", shade=True, color=colors[c], ax=axes[i,j]) esta mal creo    
            ax = sns.boxplot(x=dat_filt[usadas[j]], notch=True, color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])
            
            if (i==k-1):
                axes[i,j].set_xlabel(usadas[j])
            else:
                axes[i,j].set_xlabel("")
            
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i+1))
            else:
                axes[i,j].set_ylabel("")
            
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', visible=False)
            
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

    fig.set_size_inches(15,15)
    fig.savefig("img/" + caso + "/" + alg + "/boxplots.pdf")
    plt.clf()

def mds(centers, alg, size, k):
    print("---------- MDS...")
    
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)
    mds = MDS(random_state=123456)
    centers_mds = mds.fit_transform(centers)
    fig=plt.figure(4)
    plt.scatter(centers_mds[:,0], centers_mds[:,1], s=size*10, alpha=0.75, c=colors)
    for i in range(k):
        plt.annotate(str(i+1),xy=centers_mds[i],fontsize=18,va='center',ha='center')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.set_size_inches(15,15)
    plt.savefig("img/" + caso + "/" + alg + "/mds.pdf")
    plt.clf()

def dendrograma(X_alg, alg, usadas):
    print("---------- Dendrogramas...")
    
    sns.set_theme(color_codes=True)
    X_normal = preprocessing.normalize(X_alg, norm='l2')
    linkage_array = hierarchy.ward(X_normal)
    p = 15
    hierarchy.dendrogram(linkage_array, orientation='left', p=p, truncate_mode='lastp')
    plt.savefig("img/" + caso + "/" + alg + "/dendrograma_" + str(p) + ".pdf")
    plt.clf()
    
    X_normal = pd.DataFrame(X_normal,index=X_alg.index,columns=usadas)
    sns.clustermap(X_normal, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

    plt.savefig("img/" + caso + "/" + alg + "/dendrogramaHeatmap.pdf")
    plt.clf()

def distribucion(X_alg, alg, usadas, centers, k, n_var):
    print("---------- Distribución por variable y cluster...")
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)
    plt.figure()
    mpl.style.use('default')
    fig, axes = plt.subplots(k, n_var, sharey=True,figsize=(15,15))
    fig.subplots_adjust(wspace=0,hspace=0)


    centers_sort = centers.sort_values(by=[usadas[0]]) #ordenamos por renta para el plot

    rango = []
    for j in range(n_var):
        rango.append([X_alg[usadas[j]].min(),X_alg[usadas[j]].max()])

    for i in range(k):
        c = centers_sort.index[i]
        dat_filt = X_alg.loc[X_alg['cluster']==c]
        for j in range(n_var):
            #ax = sns.kdeplot(x=dat_filt[usadas[j]], label="", shade=True, color=colors[c], ax=axes[i,j])
            ax = sns.histplot(x=dat_filt[usadas[j]], label="", color=colors[c], ax=axes[i,j], kde=True) # mejor si se usa weights de 'DB090'
            #ax = sns.boxplot(x=dat_filt[usadas[j]], notch=True, color=colors[c], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])
            
            ax.set(xlabel=usadas[j] if (i==k-1) else '', ylabel='Cluster '+str(c+1) if (j==0) else '')
            
            ax.set(yticklabels=[])
            ax.tick_params(left=False)
            ax.grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            ax.grid(axis='y', visible=False)
            
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

    fig.set_size_inches(15,15)
    fig.savefig("img/" + caso + "/" + alg + "/distribucion.pdf")
    plt.clf()


def intercluster(alg, centers, size, k):
    print("---------- Distancia intercluster...")
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)

    fig = plt.figure()
    mpl.style.use('default')

    mds = MDS(random_state=123456)
    centers_mds = mds.fit_transform(centers)

    plt.scatter(centers_mds[:,0], centers_mds[:,1], s=size**1.6, alpha=0.75, c=colors) # mejor si se usa weights de 'DB090' para size
    for i in range(k):
        plt.annotate(str(i+1),xy=centers_mds[i],fontsize=18,va='center',ha='center')
    xl,xr = plt.xlim()
    yl,yr = plt.ylim()
    plt.xlim(xl-(xr-xl)*0.13,xr+(xr-xl)*0.13)
    plt.ylim(yl-(yr-yl)*0.13,yr+(yr-yl)*0.13)
    plt.xticks([])
    plt.yticks([])
    fig.set_size_inches(15,15)
    plt.savefig("img/" + caso + "/" + alg + "/intercluster.pdf")
    plt.clf()


def parallel_coordinates(X, clusters, metric_SC_samples, k, alg, usadas):
    print("---------- Parallel coordinates...")
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)

    plt.figure()
    mpl.style.use('default')

    X['cluster'] = clusters
    X['SC'] = metric_SC_samples
    df = X

    # si se desea aclarar la figura, se pueden eliminar los objetos más lejanos, es decir, SC < umbral, p.ej., 0.3
    df = df.loc[df['SC']>=0.3]

    colors_parcoor = [(round((i//2)/k+0.2*(i%2),3),'rgb'+str(colors[j//2])) for i,j in zip(range(2*k),range(2*k))]

    fig = px.parallel_coordinates(df, dimensions=usadas,
                                color="cluster", range_color=[-0.5, 4.5],
                                color_continuous_scale=colors_parcoor)
    
    tickvals = []
    ticktext = []
    
    for i in range(k):
        tickvals.append(k)
        ticktext.append("Cluster " + str(k))
    
    

    fig.update_layout(coloraxis_colorbar=dict(
        title="Clusters",
        tickvals=tickvals,
        ticktext=ticktext,
        lenmode="pixels", len=500,
    ))

    fig.write_html("img/" + caso + "/" + alg + "/parallel.html")
    plt.clf()


def analisisParametrosClusters(algoritmo, X, X_normal, min, max):
    # Crear directorio si no existe
    try:
        os.stat("img/" + caso)
    except:
        os.mkdir("img/" + caso)   
    
    try:
        os.stat("img/" + caso + "/" + algoritmo)
    except:
        os.mkdir("img/" + caso + "/" + algoritmo)

    try:
        os.stat("img/" + caso + "/" + algoritmo + "/parametros")
    except:
        os.mkdir("img/" + caso + "/" + algoritmo + "/parametros")

    metric_CH_vector = []
    metric_SC_vector = []

    for i in range(min, max):
        print('----- Ejecutando ' + algoritmo + ', numero de clusters: ' + str(i),end='')
        
        if algoritmo == "k-Means":
            alg = KMeans(init='k-means++', n_clusters=i, n_init=i, random_state=123456)
        elif algoritmo == "AgglomerativeClustering":
            alg = AgglomerativeClustering(n_clusters=i, linkage="ward")

        # Tiempo  
        t = time.time()
        cluster_predict = alg.fit_predict(X_normal,subset['DB090']) # se usa DB090 como peso para cada objeto (factor de elevación)
        tiempo = time.time() - t
        print(": {:.2f} segundos, ".format(tiempo), end='')

        # Calinski-Harabasz
        try:
            metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
            metric_CH_vector.append(metric_CH)
            print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
        except:
            metric_CH = -1

        # Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
        # Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
        muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
        
        # Silhouette
        try:
            metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)
            metric_SC_vector.append(metric_SC)
            print("Silhouette Coefficient: {:.5f}".format(metric_SC))
        except:
            metric_SC = -1

        # se convierte la asignación de clusters a DataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

        # Tamaño de cluster
        print("Tamaño de cada cluster:")
        size=clusters['cluster'].value_counts()
        for num,i in size.iteritems():
            print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        size = size.sort_index()
        k = len(size)

    # Gráficas
    fig, ax = plt.subplots()
    x = np.arange(min,max)
    ax.scatter(x, metric_SC_vector, marker='o')
    ax.legend(['Silhouette'])
    fig.savefig("img/" + caso + "/" + algoritmo + "/parametros" + "/silhouette.pdf")
    plt.clf()

    fig, ax = plt.subplots()
    x = np.arange(min,max)
    ax.scatter(x, metric_CH_vector, marker='o')
    ax.legend(['Calinski-Harabasz'])
    fig.savefig("img/" + caso + "/" + algoritmo + "/parametros" + "/calinski-Harabasz.pdf")
    plt.clf()

def analisisParametrosBirch(X, X_normal):
    metric_CH_vector = []
    metric_SC_vector = []

    branching_factor_vector = [10, 15, 20, 25, 30, 35, 40]
    threshold_vector = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    for i in branching_factor_vector:
        metric_CH_vector.clear()
        metric_SC_vector.clear()

        for j in threshold_vector:
            print('\n----- Ejecutando Birch, numero de clusters: 5, branching_factor: ' + str(i) + ', threshold: ' + str(j),end='')
            
            alg = Birch(branching_factor=i, threshold=j, n_clusters=5)

            # Tiempo  
            t = time.time()
            cluster_predict = alg.fit_predict(X_normal,subset['DB090']) # se usa DB090 como peso para cada objeto (factor de elevación)
            tiempo = time.time() - t
            print(": {:.2f} segundos, ".format(tiempo), end='')

            # Calinski-Harabasz
            try:
                metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
                metric_CH_vector.append(metric_CH)
                print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
            except:
                metric_CH = -1

            # Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
            # Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
            muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
            
            # Silhouette
            try:
                metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)
                metric_SC_vector.append(metric_SC)
                print("Silhouette Coefficient: {:.5f}".format(metric_SC))
            except:
                metric_SC = -1

            # se convierte la asignación de clusters a DataFrame
            clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

            # Tamaño de cluster
            print("Tamaño de cada cluster:")
            size=clusters['cluster'].value_counts()
            for num,i in size.iteritems():
                print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
            size = size.sort_index()
            k = len(size)


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    datos = pd.read_csv('datos_hogar_2020.csv')
    
    '''
    for col in datos:
    missing_count = sum(pd.isnull(datos[col]))
    if missing_count > 0:
        print(col,missing_count)
    #'''

    # Se pueden reemplazar los valores desconocidos por un número
    # datos = datos.replace(np.NaN,0)

    # O imputar, por ejemplo con la media      
    for col in datos:
        if col != 'DB040':
            datos[col].fillna(datos[col].mean(), inplace=True)

    # Seleccionar variables de interés para clustering
    # renombramos las variables por comodidad
    if caso == "caso1":
        subset = datos.loc[(datos['HY020'] > 0) & (datos['HY020'] < 150000)] # renta entre 0 y 150.000
        subset=subset.rename(columns={"HY020": "renta_neta", "HY140G": "impuesto", "HH070": "gastos_vivienda", "HH031": "año_compra", "HC010": "alimentacion_in"})
        usadas = ['renta_neta', 'impuesto', 'gastos_vivienda', 'año_compra', 'alimentacion_in']
    elif caso == "caso2":
        subset = datos.loc[(datos['HY060N'] > 0)] # reciben asistencia social
        subset=subset.rename(columns={"HY060N": "asistencia_social", "HY020": "renta", "HC010": "alimentacion_in", "HS130": "ingresos_min"})
        usadas = ['asistencia_social', 'renta', 'alimentacion_in', 'ingresos_min']
    elif caso == "caso3":
        subset = datos.loc[(datos['HY030N'] > 0)] # alquiler imputado
        subset=subset.rename(columns={"HY030N": "alquiler_imputado", "HY020": "renta", "HC010": "alimentacion_in", "HH061": "importe_pensado", "HC040": "transporte_privado"})
        usadas = ['alquiler_imputado', 'renta', "alimentacion_in", 'importe_pensado', 'transporte_privado']

    n_var = len(usadas)
    X = subset[usadas]

    # eliminar outliers como aquellos casos fuera de 1.5 veces el rango intercuartil
    X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
    # normalizamos
    X_normal = X.apply(norm_to_zero_one)
    
    # algoritmos
    k_means = KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=123456)
    mean_shift = MeanShift(bandwidth=0.3)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=5, linkage="ward")
    dbscan = DBSCAN(eps=0.14, min_samples=5)
    birch = Birch(branching_factor=25, threshold=0.25, n_clusters=5)

    algoritmos = {'k-Means' : k_means, 'MeanShift' : mean_shift, 'Birch' : birch, 'AgglomerativeClustering' : agglomerative_clustering, 'DBSCAN' : dbscan}
    
    for alg in algoritmos:
        print('----- Ejecutando ' + alg,end='')
        
        # Crear directorio si no existe
        try:
            os.stat("img")
        except:
            os.mkdir("img")

        try:
            os.stat("img/" + caso)
        except:
            os.mkdir("img/" + caso)   
        
        try:
            os.stat("img/" + caso + "/" + alg)
        except:
            os.mkdir("img/" + caso + "/" + alg)   
          
        # Tiempo  
        t = time.time()
        cluster_predict = algoritmos[alg].fit_predict(X_normal,subset['DB090']) # se usa DB090 como peso para cada objeto (factor de elevación)
        tiempo = time.time() - t
        print(": {:.2f} segundos, ".format(tiempo), end='')

        # Calinski-Harabasz
        try:
            metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
            print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
        except:
            metric_CH = -1
        # Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
        # Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
        muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
        
        # Silhouette
        try:
            metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)
            print("Silhouette Coefficient: {:.5f}".format(metric_SC))
        except:
            metric_SC = -1
            
        try:
            metric_SC_samples = metrics.silhouette_samples(X_normal, cluster_predict, metric='euclidean')
        except:
            metric_SC_samples = -1
        # se convierte la asignación de clusters a DataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

        # Tamaño de cluster
        print("Tamaño de cada cluster:")
        size=clusters['cluster'].value_counts()

        size = size.sort_index()
        tam = []
        n = []
        for num,i in size.iteritems():
            n.append(num)
            tam.append(i)
            print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

        
        fig, ax = plt.subplots()
        ax.bar(n, tam, color='green')
        fig.savefig("img/" + caso + "/" + alg + "/tamanio_clusters.pdf")
        plt.clf()

        k = len(size)
        
        # Centros
        X_normal_alg = pd.concat([X_normal, clusters], axis=1)
        cluster_centers = X_normal_alg.groupby("cluster").mean()
        centers = pd.DataFrame(cluster_centers, columns=list(X))
        
        # se añade la asignación de clusters como columna a X
        X_alg = pd.concat([X, clusters], axis=1)
        
        # Gráficas
        heatmap(X, alg, centers)
        
        if alg == "AgglomerativeClustering":
            dendrograma(X, alg, usadas)  
        
        try:
            scatter_matrix(X_alg, alg, k)
        except:
            pass
        
        try:
            boxplot(X_alg, usadas, centers, alg, k, n_var)
        except:
            print("ERROR.")
        
        try:
            distribucion(X_alg, alg, usadas, centers, k, n_var)
        except:
            print("ERROR.")
        
        try:
            intercluster(alg, centers, size, k)
        except:
            print("ERROR.")
        
        try:
            parallel_coordinates(X.copy(), clusters, metric_SC_samples, k, alg, usadas)
        except:
            print("ERROR.")
        
        try:
            mds(centers, alg, size, k)
        except:
            print("ERROR. mds")
            
    # Analisis de parametros
    if caso == "caso1":
        analisisParametrosClusters("k-Means", X.copy(), X_normal.copy(), 2, 21)
        analisisParametrosBirch(X.copy(), X_normal.copy())
    elif caso == "caso2" or caso == "caso3":
        analisisParametrosClusters("k-Means", X.copy(), X_normal.copy(), 2, 21)
        analisisParametrosClusters("AgglomerativeClustering", X.copy(), X_normal.copy(), 2, 21)