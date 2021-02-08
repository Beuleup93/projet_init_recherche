#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:57:45 2020

@author: macbookair
"""
# Modification du dossier par défaut
import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.cache import Cache


import pandas as pd
import igraph
import numpy as np
import h5py
from collections import Counter
import io


class OperatorWindow(BoxLayout):
    #chemin = '/Users/macbookair/Desktop/Lyon2/ProjetGroup/ProjetAnalyse/operator/graphe.png'
    repertoire = '/Users/macbookair/Desktop/Lyon2/ProjetGroup/ProjetAnalyse/operator'
    methodeCommunaute = ''
    graph = igraph.Graph()
    sortedBy = ''
    taille = ''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coauteurs = pd.read_csv('/Users/macbookair/Desktop/Lyon2/ProjetGroup/ProjetAnalyse/dataset/coauteur.csv',header=0)
        self.dataframe = pd.read_csv('/Users/macbookair/Desktop/Lyon2/ProjetGroup/ProjetAnalyse/dataset/dataframe.csv',header=0)
        del(self.coauteurs['Unnamed: 0'])
        A = np.ndarray(shape=(0,0),dtype=int)

    def update_purchases(self,value):
        OperatorWindow.taille = value
        self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n'
        self.ids.receipt_preview.text += 'Echantillon: '+ value+' auteurs\n\n'
        self.loadFile(int(value))

    def getCoauteurs(self):
        return self.coauteurs

    def getDataframe(self,taille):
        return self.dataframe.head(taille)
        
    def getA(self):
        return self.A

    def setA(self, matrice):
        self.A = matrice

    def get_adjacence_matrice_file(self):
        if OperatorWindow.taille.isdigit() and len(OperatorWindow.taille)>0:
            f2 = h5py.File('MatriceAdj_'+OperatorWindow.taille+'.hdf5', 'r')
            #Afficher les noms des datasets dans le fichier hdf5:
            #list(f2.keys())
            # Récupérer les données:
            dset1 = f2['MatriceAdj_'+OperatorWindow.taille]
            data = dset1[:]
            return data
        else:
            #print("Veuillez choisir correctement la taille de l'echantillon pour contruire la matrice d'adjacence")
            self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n Veuillez choisir la taille de l\'echantillon pour creer la matrice d\'adjacence'
            return np.ndarray(shape=(0,0),dtype=int)
            
    def get_adjacence_matrice(self):
        #self.ids.images.clear_widgets()
        if OperatorWindow.taille.isdigit() and len(OperatorWindow.taille)>0:
            df = self.getDataframe(int(OperatorWindow.taille))
            dfRelation = self.getCoauteurs()
            self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n Création de la matrice de taille '+str(OperatorWindow.taille)+'*'+str(OperatorWindow.taille)
            matrice = self.matriceAdjacence(df,int(OperatorWindow.taille))
            self.ids.receipt_preview.text += "\n\n_____********************************____\n\n Matrice d'adjacence créée"
            return matrice
        else:
            print("Veuillez choisir correctement la taille de l'echantillon pour contruire la matrice d'adjacence")
            return np.ndarray(shape=(0,0),dtype=int)

    def construct_matrice(self):
            self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n Création de la matrice de taille '+str(OperatorWindow.taille)+'*'+str(OperatorWindow.taille)
            self.ids.receipt_preview.text += "\n\n_____********************************____\n\n Matrice d'adjacence créée"
       
            self.setA(self.get_adjacence_matrice_file())
           

    def loadFile(self,taille):
        coauteur = self.getCoauteurs()
        #del(coauteur['Unnamed: 0'])
        dataframe = self.getDataframe(int(taille))
        
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        coauteur.info(buf=buf1)
        dataframe.info(buf=buf2)
        s1 = buf2.getvalue()
        s2 =  buf1.getvalue()
        
        self.ids.receipt_preview.text += ' - Fichier des auteurs \n\n'+s1+ '\n\n - Fichier de relation entre auteurs\n\n' +s2
    
  # Verification de l'existence d'une relation entre 2 auteurs
    def isRelationshipWithAuthor(self,dfRelation, idx, idy):
        
        relationship = dfRelation.loc[(dfRelation['id_author_x']==idx) & (dfRelation['id_author_y']==idy),:]['id_publication'].count()
        
        if relationship == 0:
            return False
        
        else:
            return True
        
    def matriceAdjacence(self,dataframe,dfRelation, n):
    
        matrice = np.ndarray(shape=(n,n), dtype = int)
        
        print (" Construction de la matrice carré de taille (%d*%d) "%(n,n))
        
        for i in range(n):
            
            for j in range(n):
                
                isRelation = self.isRelationshipWithAuthor(dfRelation, dataframe.iloc[i,0], dataframe.iloc[j,0])
                
                if isRelation:
                    matrice[i][j] = 1
                    
                else:
                    matrice[i][j] = 0
                    
        return matrice

    def show_graphe(self):
        self.reset_all()
        if self.getA().shape[0] != 0:
            '''
            g = igraph.Graph.Adjacency(self.A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
            OperatorWindow.graph = g
            g.vs["label"] = self.getDataframe(self.A.shape[0]).index.tolist()
            ob = igraph.plot(g,vertex_label_size=15,vertex_size=25,vertex_color='green') 

            if OperatorWindow.taille == '35' and os.path.exists('graphe/graphe35.png') is False:
                ob.save('graphe/graphe35.png')
            elif OperatorWindow.taille == '50' and os.path.exists('graphe/graphe50.png') is not True:
                ob.save('graphe/graphe50.png')
            elif OperatorWindow.taille == '100' and os.path.exists('graphe/graphe100.png') is False:
                ob.save('graphe/graphe100.png')
            elif OperatorWindow.taille == '150' and os.path.exists('graphe/graphe150.png') is False:
                ob.save('graphe/graphe150.png')
            elif OperatorWindow.taille == '200' and os.path.exists('graphe/graphe200.png') is False:
                ob.save('graphe/graphe200.png')'''
            # Show Graphe
            if os.path.exists('graphe/graphe'+OperatorWindow.taille+'.png') is True:
                self.show_communaute('graphe/graphe'+OperatorWindow.taille+'.png')
        else:
            print("Veuiller Renseigner la taille de l'echantillon et puis construire la matrice")

    def getCommunauteByMethode(self, graphe,typeAlgo):
        #Vider le container
        self.reset_all()
        # Get Methode
        community = []
        if typeAlgo == 'Multi_Level':
            print("Multi_level")
            community = graphe.community_multilevel (weights = None , return_levels = False )
            # Structure de la communauté basée sur l'algorithme multi-niveaux de Blondel et al
        elif typeAlgo == 'Newman':
            print('Newman')
            community = graphe.community_leading_eigenvector(clusters = None , weights = None , arpack_options = None )
        elif typeAlgo == 'Between':
            print('Edge Between')
            community = graphe.community_label_propagation (weights = None , initial = None , fixed = None )
            #community = graphe.community_edge_betweenness(directed=False)
            #print((community.as_clustering(n=community.optimal_count)))
        else:
            print("Multi_level")
            community = graphe.community_multilevel (weights = None , return_levels = False )
           
        print("community",len(community))
        print(community)
        '''
        communitiesImage = []
        #sortedCommunityMaxInd = self.sortedCommunities(community)
        for i in range(len(community)):
            #if len(community[i]) > 3 :
                cluster = graphe.vs.select(graphe.blocks()[i])
                ob = igraph.plot(graphe.subgraph(cluster),vertex_label_size=15,vertex_size=25,vertex_color='green') 
                #if os.path.exists('communaute/'+OperatorWindow.taille+'/graphe'+str(i)+'.png') is False:
                ob.save('communaute/'+OperatorWindow.methodeCommunaute+'/'+OperatorWindow.taille+'/'+str(i)+'.png')
                    #communitiesImage.append('communaute/'+OperatorWindow.taille+'/graphe'+str(i)+'.png')
        '''
        # Afficher les communauté
        self.afficher_communaute(community)

        for i in range(6): # Affichage 6 communauté
            self.show_communaute('communaute/'+OperatorWindow.methodeCommunaute+'/'+OperatorWindow.taille+'/'+str(i)+'.png')

    def show_communaute(self, chemin):
        images_container = self.ids.images
        image = Image(source = chemin,
                      size_hint=(1, None),
                      size_hint_y=.05,
                      keep_ratio=True,
                      allow_stretch=True)
        #Cache.remove('image.kv')
        images_container.add_widget(image)

    def reset_all(self):
        self.ids.images.clear_widgets()
        self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n'
        #self.remove_img('graphe.png')

    def remove_img(self,img_name):
        if os.path.exists(img_name) is True:
            print(os.path.exists(img_name))
            os.remove(img_name)
            return True

    def spinner_clicked(self, value): 
        OperatorWindow.methodeCommunaute = value
        print("Valeur choisie " + value)

    def spinner_sorting(self, value): 
        OperatorWindow.sortedBy = value
        print("Sorting " + value)

    def detecter_communaute(self):
        A = self.getA()
        print(A)
        g = igraph.Graph.Adjacency(A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
        g.vs["label"] = self.getDataframe(A.shape[0]).index.tolist()
        g.vs["name"] = self.getDataframe(A.shape[0]).name_author.tolist()
        print(g)
        self.getCommunauteByMethode(g,OperatorWindow.methodeCommunaute)

    def sortAuthorByAttr(self,graphe,attr,taille):
        m=sorted(graphe.vs, key=lambda x:x[attr], reverse=True)
        self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n \t\t\t\t\t\t Correspondace Index - Name - (40):\n\n'
        for e in m[:taille]:
            self.ids.receipt_preview.text +=str(e['label']) +'....'+ e['name']+'\n'

    def auteur_centraux(self):

        A = self.getA()
        g = igraph.Graph.Adjacency(self.A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
        g.vs["label"] = self.getDataframe(A.shape[0]).index.tolist()
        g.vs["name"] = self.getDataframe(A.shape[0]).name_author.tolist()

        self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n \t\t\t\t\t\t Auteurs centraux (10) \n\n'
        
        # 10 premiers sommets plus important, tri de manière décroissante(en terme de voisinage)
        self.ids.receipt_preview.text += '--- En terme de voisinage ---\n\n'
        valeurs = {'node_label':g.vs['label'],'node_name':g.vs['name'],'degree':g.vs.degree()}
        df1 = pd.DataFrame.from_dict(valeurs).sort_values(by='degree',ascending=False).iloc[:10,:]
        maliste1 = df1.values.tolist()
        print(maliste1)
        for i in range(len(maliste1)):
            self.ids.receipt_preview.text +=str(maliste1[i][0])+'...'+maliste1[i][1]+'...'+str(maliste1[i][2])+'\n'

        #closeness centralité (en terme de distance)
        self.ids.receipt_preview.text += '\n\n'
        self.ids.receipt_preview.text += '---En terme de distance ----\n\n'
        valeurs = {'node_label':g.vs['label'],'node_name':g.vs['name'],'closeness':g.vs.closeness()}
        df2 = pd.DataFrame.from_dict(valeurs).sort_values(by='closeness',ascending=False).iloc[:10,:]
        maliste2 = df2.values.tolist()
        print(maliste2)
        for i in range(len(maliste2)):
            self.ids.receipt_preview.text +=str(maliste2[i][0])+'...'+maliste2[i][1]+'...'+str(maliste2[i][2])+'\n'


        # betweenness centralité (plus court chemin)
        self.ids.receipt_preview.text += '\n\n'
        self.ids.receipt_preview.text += '--- En Fonction du plus court chemin ---\n\n'
        valeurs = {'node_label':g.vs['label'],'node_name':g.vs['name'],'betweenness':g.vs.betweenness()}
        df3 = pd.DataFrame.from_dict(valeurs).sort_values(by='betweenness',ascending=False).iloc[:10,:]
        maliste3 = df3.values.tolist()
        print(maliste3)
        for i in range(len(maliste3)):
            self.ids.receipt_preview.text +=str(maliste3[i][0])+'...'+maliste3[i][1]+'...'+str(maliste3[i][2])+'\n'


        #print(pd.DataFrame.from_dict(valeurs).sort_values(by='betweenness',ascending=False).iloc[:5,:])

    def afficher_communaute(self,Community):
        A = self.getA()
        g = igraph.Graph.Adjacency(self.A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
        g.vs["name"] = self.getDataframe(A.shape[0]).name_author.tolist()
        g.vs["label"] = self.getDataframe(A.shape[0]).index.tolist()
        g.vs["between"] = g.betweenness()

        self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n \t Affichage des '+(str(len(Community)))+' Communaté d\'Auteurs Détecter :\n\n'
        self.ids.receipt_preview.text += '-------------- COMMUNAUTE 0 ---------------- \n\n'
        for i in range(len(Community)):
            for z in Community[i]:
                self.ids.receipt_preview.text += str(g.vs[z]['label'])+'...'+g.vs[z]['name']+'...'+str(g.vs[z]['between'])+'\n'

            self.ids.receipt_preview.text += '-------------- COMMUNAUTE '+str(i)+' -------------\n\n'

    def find_degree(self,g,sommet):
        return g.vs.degree

    def auteur_voisinage(self):
        voisinage = self.ids.voisinage_inp.text
        A = self.getA()
        g = igraph.Graph.Adjacency(self.A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
        g.vs["name"] = self.getDataframe(A.shape[0]).name_author.tolist()
        g.vs["label"] = self.getDataframe(A.shape[0]).index.tolist()

        if voisinage.isdigit() and len(voisinage)>0:
            self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n \t\t\t\t Les Voisinages de ('+voisinage+')\n\n'

            self.ids.receipt_preview.text += '\n **** Noms Auteurs **** \n\n'
            for v in g.vs[g.neighborhood(int(voisinage))]["name"]:
                self.ids.receipt_preview.text += v+'\n'
            
            self.ids.receipt_preview.text += '\n **** Sommets **** \n\n'
            for v in g.vs[g.neighborhood(int(voisinage))]["label"]:
                self.ids.receipt_preview.text += str(v)+'\n'
        elif voisinage=='fort':
            # Calcul des edges betweens de chaque sommets
            edge_bet = g.edge_betweenness() 
            print(edge_bet)
            # Identifier sommet le plus fort
            # Transformons en vecteur numpy
            ebn = np.array(edge_bet)
            print(ebn)
            # Indice de la valeur maximale
            iebn = np.argmax(ebn)
            self.ids.receipt_preview.text = '\t\t\t\t Main Screen \n\n \t\t\t\t Sommet le plus fort \n\n'
            self.ids.receipt_preview.text += str(iebn)+'--------'+(g.vs[iebn]["name"])


    def voisinage(self):
        voisinage = self.ids.voisinage_inp.text
        self.ids.receipt_preview.text = '\t\t\t\t\t Main Screen \n\n \t\t\t\t\t Les Voisinages de ('+voisinage+') :\n\n'
    
    def create_graphe(self):
        A = self.getA()
        if A.shape !=0 : 
            g = igraph.Graph.Adjacency(self.A.tolist(),mode=igraph.ADJ_UNDIRECTED) 
            g.vs["name"] = self.getDataframe(A.shape[0]).name_author.tolist()
            g.vs["label"] = self.getDataframe(A.shape[0]).index.tolist()
            g.vs["between"] = g.betweenness()
            return g


    def conclusion(self):
        self.show_communaute('res.png')

class OperatorApp(App): 
    def build(self): 
        return OperatorWindow()

if __name__ == "__main__":
    oa = OperatorApp()
    oa.run()
