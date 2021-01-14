import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
class KMean():
    def __init__(self, X, max_iters=50, plot_steps=False, init='kmeans++'):
        self.X=X
        self.n,self.d = X.shape
        self.max_iters=max_iters
        self.plot_steps = plot_steps
        self.initial = init

    # nhom kmean
        # lay k tam nhau nhien
    def kmean_random_centroids(self,k): 
        return self.X[np.random.choice(self.n,k,replace=False)]
    
        # lay k tam theo thuat toan Kmeans++ 
    def kmean_plus_plus(self,k):
        centroids_temp = []
        centroids_temp.append(self.X[np.random.choice(self.n)])
        for _ in range(k-1):
            dist= []
            for i in range(self.n):
                data = self.X[i,:]
                temp_dist=[]
                for centroid in centroids_temp:
                    temp_dist.append(np.sum((data - centroid)**2))
                dist.append(np.min(temp_dist))
            centroids_temp.append(self.X[np.argmax(dist),:])
        centroids_temp = np.array(centroids_temp)
        return centroids_temp
        
        # Tinh khoang cach Euclid tu cac diem den cac tam
    def kmean_euclid_dis(self,k):
        distances=np.zeros((self.n,k))
        for i in range(k):
            for j in range(self.d):
                distances[:,i]+=(self.X[:,j]-self.centroids[i,j])**2
        distances=np.sqrt(distances)
        return distances
    
        # lay nhan cua cac diem gan voi cac tam nhat
    def kmean_assign_labels(self,k):
        distances = self.kmean_euclid_dis(k)
        return np.argmin(distances,axis=1)
    
        # tinh lai vi tri cac tam
    def kmean_update_centroids(self,k):
        return np.array([np.mean(self.X[self.labels == i,:],axis=0) for i in range(k)])
    
        # Kiem tra xem 2 tam co trung nhau khong
    def kmean_check_centroids(self, new_centroids):
        return (set([tuple(i) for i in self.centroids])==set([tuple(i) for i in new_centroids])) # Vu Huu Tiep book
    
        # Tao cac Cluster
    def kmean_create_clusters(self):
        self.clusters = [self.X[self.labels == i,:] for i in range(len(self.centroids))]
        return self.clusters
    
        # Plot kmean
    def kmean_plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        defi_color=['b','g','r','c','m','y','k','b','g','r','c','m','k','w','b','g','r','b','g','r','c','m','y','k','b','g','r','c','m','k','w','b','g','r','b','g','r','c','m','y','k','b','g','r','c','m','k','w','b','g','r']
#         defi_center_color=['m','y','k','b','g','r','c','m','k','b','g','r','c','w','b','g','r']
        for i in range(len(self.centroids)):
            ax.scatter(self.clusters[i][:,0],self.clusters[i][:,1],marker='o',color=defi_color[i],alpha=0.7,s=8**2)
            ax.scatter(self.centroids[i,0],self.centroids[i,1],marker='x',color='k',s=12**2)
        plt.show()


        # Chay thuat toan Kmean
    def fit(self,k,plot_steps=True):
        if self.initial.lower() in ['random','rd']:
            self.centroids=self.kmean_random_centroids(k) # Thiet lap tam ngau nhien k tam nhau nhien ban dau
        elif self.initial == 'kmeans++':
            self.centroids=self.kmean_plus_plus(k)
        for _ in range(self.max_iters):
            
            self.labels=self.kmean_assign_labels(k)
                        
            self.clusters=self.kmean_create_clusters()
            if plot_steps and self.plot_steps:
                self.kmean_plot()
                
            new_centroids = self.kmean_update_centroids(k)
            if self.kmean_check_centroids(new_centroids):
                break
            self.centroids=new_centroids

        
        return self.centroids, self.labels
    
    # Nhom Elbow
        # Ham elbow
    def elbow_method(self,num):
        elbow_values = np.zeros(num+1)
        elbow_values[0]=None
        for i in range(1,num+1):
            Centroids,labels=self.fit(i,plot_steps=False)
            for j in range(i):
                elbow_values[i]+=np.sum((self.X[labels==j,:]-Centroids[j])**2)
        self.elbow_values=elbow_values
        return elbow_values
        
        # Hien thi cac gia tri cua elbow method
    def Print_elbow_method(self):
        try:
            for i in range(1,len(self.elbow_values)):
                print('k={}:{}'.format(i,self.elbow_values[i]))
        except :
            print("No values in elbow method yet!")
        
        # Bieu do hoa cac gia tri cua elbow method
    def Show_elbow_method(self):
        try:
            plt.plot(self.elbow_values,'or-')
            plt.show()
        except:
            print("No values in elbow method yet!")
    # Nhom Silhouette
        # Ham silhoutte
    def silhouette_method(self,num,Test=True):
        self.sild_num = num
        Silhoue_mean=np.zeros(num+1)
        Silhoue_mean[0]=Silhoue_mean[1]=None
        Silh_k=[]
        silh_kmean_values=[]
        if Test == True:
            for k in range(2,num+1):
                X_centroids_s, labels_s=self.fit(k, plot_steps=False)
                silh_kmean_values.append([X_centroids_s, labels_s])
                Xlabels=np.concatenate((self.X,labels_s.reshape(-1,1)),axis=1)
                silh=np.zeros((self.n,2))
                for index in range(self.n):
                    Si=self.X[index]
                    #Tách các cụm.
#                        print(Xlabels,Xlabels[index,self.d])
                    Atemp = Xlabels[Xlabels[:,self.d]==Xlabels[index,self.d]]
                    Btemp = Xlabels[Xlabels[:,self.d]!=Xlabels[index,self.d]]
#                     print('Btemp',index,Btemp[:1])
                    #Chọn một giá trị ngẫu nhiên trong cụm và loại bỏ giá trị đã chọn trong cụm đó.
                    if Si in Atemp:
                        Atemp=np.delete(Atemp,np.where(Atemp==Si),axis=0)
                    if Si in Btemp:
                        Btemp=np.delete(Btemp,np.where(Btemp==Si),axis=0)
        
                    if (Atemp.shape[0]-1)==0: # Nếu số lượng phần tử trong cụm = 1 thì trả về 0 để tránh trường hợp chia cho 0.
                        silh[Si]=0
                        continue
                    # Tổng khoảng cách từ i đến các phần tử j trong cụm
                    SumSa=0
                    for j in range(Atemp.shape[1]-1):
                        SumSa+=(Si[j]-Atemp[:,j])**2

                    SumSa=np.sum(np.sqrt(SumSa))

                    SA = (1/(Atemp.shape[0]-1))*SumSa
                    # Tổng khoảng cách từ i đến các phần tử không thuộc cụm i
                    othercl=np.delete(np.array(range(k)),int(Xlabels[index,self.d]))
#                     print("othercl",index,othercl[:1])
                    TempBtemp =np.zeros(k-1) # Biến lưu các khoảng cách để chọn giá trị thấp nhất.
                    for l in range(k-1):
                        SumSb = 0
                        Temp1=Btemp[Btemp[:,self.d]==othercl[l]] # lấy các phần tử thuộc các cụm khác cụm của i.
#                             print(Temp1)
                        for j in range(Temp1.shape[1]-1):
                            SumSb+=(Si[j]-Temp1[:,j])**2
                        SumSb=np.sum(np.sqrt(SumSb))
                        TempBtemp[l]=(1/Temp1.shape[0])*SumSb
                    SB = np.amin(TempBtemp,axis=0)
                    silh[index]=[(SB-SA)/(np.max([SB,SA])),Xlabels[index,self.d]]
#                     print('silh[{}]:{}'.format(index,silh[index]))
                Silh_k.append(silh)
                Silhoue_mean[k]=np.mean(silh[:,0])
            self.Silhoue_mean=Silhoue_mean
            self.Silh_k_values=Silh_k
            self.Silh_kmean_values= silh_kmean_values
            return Silh_k
        else:
            for k in range(2,num+1):
                X_centroids_s, labels_s=self.fit(k, plot_steps=False)
                Xlabels=np.concatenate((self.X,labels_s.reshape(-1,1)),axis=1)
                silh=np.zeros(k)
                for i in range(k):
                    #Tách các cụm.
                    Atemp = Xlabels[Xlabels[:,self.d]==i]
                    Btemp = Xlabels[Xlabels[:,self.d]!=i]
                    #Chọn một giá trị ngẫu nhiên trong cụm và loại bỏ giá trị đã chọn trong cụm đó.
                    rd_values=rd.randint(0,Atemp.shape[0]-1)
                    Si=Atemp[rd_values]
                    Atemp=np.delete(Atemp,rd_values,axis=0)
                    if (Atemp.shape[0]-1)==0: # Nếu số lượng phần tử trong cụm = 1 thì trả về 0 để tránh trường hợp chia cho 0.
                        silh[i]=0
                        continue
                    # Tổng khoảng cách từ i đến các phần tử j trong cụm
                    SumSa=0
                    for j in range(Atemp.shape[1]-1):
                        SumSa+=(Si[j]-Atemp[:,j])**2

                    SumSa=np.sum(np.sqrt(SumSa))

                    SA = (1/(Atemp.shape[0]-1))*SumSa
                    # Tổng khoảng cách từ i đến các phần tử không thuộc cụm i
                    othercl=np.delete(np.array(range(k)),i)

                    TempBtemp =np.zeros(k-1) # Biến lưu các khoảng cách để chọn giá trị thấp nhất.
                    for l in range(k-1):
                        SumSb = 0
                        Temp1=Btemp[Btemp[:,self.d]==othercl[l]] # lấy các phần tử thuộc các cụm khác cụm của i.
                        for j in range(Temp1.shape[1]-1):
                            SumSb+=(Si[j]-Temp1[:,j])**2
                        SumSb=np.sum(np.sqrt(SumSb))
                        TempBtemp[l]=(1/Temp1.shape[0])*SumSb
                    SB = np.amin(TempBtemp,axis=0)
                    silh[i]=(SB-SA)/(np.max([SB,SA]))
                Silhoue_mean[k]=np.mean(silh)
                self.silhouette_values=Silhoue_mean
            return Silhoue_mean
    
        # Hien thi cac gia tri cua silhoutte method
    def Print_silhoutte_method(self):
        try:
            for i in range(2,len(self.silhouette_values)):
                print('k={}:{}'.format(i,self.silhouette_values[i]))
        except :
            print("No values in silhoutte method yet!")
    def Show_silhoutte_method(self,s=True):
#         try:
            if s is False:
                plt.plot(self.silhouette_values,'or-')
                plt.show()
            if s is True:
                for k in range(2,self.sild_num+1):
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.set_size_inches(18, 7)
                    ax1.set_xlim([-0.1, 1])
                    ax1.set_ylim([0, self.n + (k + 1) * 10])
                    clusterer, cluster_labels = self.Silh_kmean_values[k-2] #Km3.fit(k)
                    Texx=self.Silh_k_values[k-2]
                    y_lower = 10
                    for i in range(k):
                        ith_cluster_silhouette_values = Texx[Texx[:,1]==i]
                        ith_cluster_silhouette_values= ith_cluster_silhouette_values[:,0]
                        ith_cluster_silhouette_values.sort()
                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i
                        color = cm.nipy_spectral(float(i) / k)
                        ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
                            # Label the silhouette plots with their cluster numbers at the middle
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples
                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=self.Silhoue_mean[k], color="red", linestyle="--")

                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
                    ax2.scatter(self.X[:, 0], self.X[:, 1], marker='o', s=30, lw=0, alpha=0.7,c=colors, edgecolor='k')
#         except:
#             print("No values in silhoutte method yet!")
    # Nhom Gap Statisitc 