[1](https://www.cnblogs.com/maoyiting/p/13354376.html)

树形 DP 进阶练习题：*P4516 [JSOI2018] 潜入行动，*P4099 [HEOI2013]SAO，*P3354 [IOI 2005] Riv 河流，*P3647 [APIO2014] 连珠 

树上背包复杂度：

- 合并复杂度为 $sz$ 乘积。每对点只会在它们 $\text{lca}$ 处合并一次，所以是 $\mathcal O(n^2)$ 的。

- 合并复杂度为 $\min(sz_x,k)\times \min(sz_y,k)$，等价于枚举子树 $x$ 中 dfs 序后 $k$ 个和子树 $y$ 中 dfs 序前 $k$ 个。考虑单个点的贡献，只会和 dfs 序相邻的左右 $2k$ 个点转移并产生 $1$ 的贡献，所以是 $\mathcal O(nk)$ 的。

##### C1

P2899 [USACO08JAN]Cell Phone Network G

> $n$ 个点的树，你要放置数量最少的信号塔，保证所有点要么有信号塔要么与信号塔相邻。求最少数量。
>
> $1\leq n\leq 10^4$。

树形 DP 大多设置子树状态，从下往上推。

每个点，被父亲/自己/儿子覆盖。设 $f_x$ 表示 $x$ 的子树，除了 $x$ 可能没覆盖，其余都被覆盖了。因为当 $x$ 子树是否放信号塔已经决策好后，$fa_x$ 只能影响 $x$ 的覆盖状态，没法覆盖 $x$ 子树内其他没覆盖的点。

设 $f_{x,0/1/2}$ 表示 $x$ 子树内部已经覆盖，$x$ 没有被信号塔覆盖/有信号塔/被信号塔覆盖（分别对应被父亲/自己/儿子覆盖），最少放多少信号塔。$f_{x,0}=\sum_{y\in son_x}\min(f_{y,1},f_{y,2})$，$f_{x,1}=1+\sum_{y\in son_x}\min(f_{y,0},f_{y,1},f_{y,2})$，$f_{x,2}=f_{y,1}+\sum_{y'\neq y}\min(f_{y',1},f_{y',2})$（一定要有个儿子覆盖它，其余随意）。

实现时：

- 方法 1：依次加入每个儿子，设现在加到 $y$，如果 $y$ 放了信号塔，$f_{x,2}=f_{y,1}+f_{x,0}$（$f_{x,0}$ 肯定比 $f_{x,2}$ 优），否则 $f_{x,2}=f_{x,2}+\min(f_{y,1},f_{y,2})$。
- 方法 2：$f_{x,2}=\min_{y\in son_x}\{f_{x,0}-\min(f_{y,1},f_{y,2})+f_{y,1}\}$。

也可以贪心：从底层的叶子，遵循只要能不放信号塔就不放。因为如果一个点可以不放但放了，它可以覆盖儿子和父亲，而可以不放意味着儿子已经被覆盖了，不如改为放它的父亲，这样有更多的点受益。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e4+5;
int n,x,y,f[N][3];
vector<int>v[N];
void dfs(int x,int fa){
	f[x][1]=1;
	for(int y:v[x]) if(y!=fa){
		dfs(y,x);
		f[x][0]+=min(f[y][1],f[y][2]);
		f[x][1]+=min({f[y][0],f[y][1],f[y][2]});
	}
	f[x][2]=1e9;
	for(int y:v[x])
		if(y!=fa) f[x][2]=min(f[x][2],f[x][0]-min(f[y][1],f[y][2])+f[y][1]);
}
signed main(){
	scanf("%d",&n);
	for(int i=1;i<n;i++)
		scanf("%d%d",&x,&y),v[x].push_back(y),v[y].push_back(x);
	dfs(1,0),printf("%d\n",min(f[1][1],f[1][2]));
	return 0;
}
```

##### C2

P1272 重建道路，同 CF440D Berland Federalization（\*2200），树上背包

> $n$ 个点的树，求最少删去的边数，使得存在 $k$ 个点的树与其他点分离。
>
> $1\leq k,n\leq 150$。

设 $f_{x,i}$ 表示保留 $x$ 子树内与 $x$ 连通的 $i$ 个点，**强制删去 $x$ 的父亲边**，最少删去的边数。

初始 $f_{x,1}=deg_x$，$f_{x,i}=\min(f_{x,i},f_{x,j}+f_{y,i-j}-2)$（$-2$：本来 $f_{y,i-j}$ 强制断开 $(x,y)$，现在不断了；$y$ 没出现时默认 $(x,y)$ 断开了，现在不断了，可以思考 $f_{x,1},f_{y,i-1}$ 的情形进行理解，还没遇到的边实际上删掉了）。

注意下标的上下界和转移顺序。时间复杂度 $\mathcal O(nk)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=160;
int n,k,x,y,f[N][N],sz[N],ans=1e9;
vector<int>v[N];
void dfs(int x,int fa){
	f[x][1]=v[x].size(),sz[x]=1;
	for(int y:v[x]) if(y!=fa){
		dfs(y,x);
		for(int i=sz[x];i>=0;i--)
			for(int j=min(sz[y],k-i);j>=0;j--)
				f[x][i+j]=min(f[x][i+j],f[x][i]+f[y][j]-2);
		sz[x]+=sz[y];
	}
}
signed main(){
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)
		scanf("%d%d",&x,&y),v[x].push_back(y),v[y].push_back(x);
	memset(f,0x3f,sizeof(f)),dfs(1,0);
	for(int i=1;i<=n;i++) ans=min(ans,f[i][k]);
	printf("%d\n",ans);
	return 0;
}
```

##### C3

P3780 [SDOI2017]苹果树，LOJ#2268. 「SDOI2017」苹果树，树上多重背包，依赖背包的小技巧

> $n$ 个节点的有根树，第 $i$ 个节点上有 $a_i$ 个苹果，取走一个苹果可以得到 $v_i$ 的幸福度。
>
> 如果在一个节点取走了至少一个苹果，则必须要在其父节点处取走至少一个苹果。
>
> 给定正整数 $k$，若取走了 $t$ 个苹果，取走至少一个苹果的节点中最深的深度为 $h$，要求 $t-h\leq k$。
>
> 问最大能收获的幸福度。
>
> $1\leq n\leq 2\times 10^4$，$1\leq k\leq 5\times 10^5$，$1\leq nk\leq 2.5\times 10^7$，$1\leq a_i\leq 10^8$，$1\leq v_i\leq 100$。

$t-h\leq k$ 的实际意义：选取一条到叶子的链免费获取（不到叶子不优），其他用树上依赖背包来付费获取，选取儿子的前提是选取父亲，背包大小为 $k$。

枚举这个叶子，剩下的就要占用背包大小了：链上剩下的部分选；链的左边选； 链的右边选。把每个点拆成一个苹果和 $a_i-1$ 个苹果的虚儿子就不存在第一种情况了，这样仍满足依赖关系，只需求出第二三种情况的背包，然后单次 $\mathcal O(k)$ 单次合并。

由于此题不是 01 背包，一个点可以取多次，转移不能通过 $sz$ 取 $\min$，故转移合并是 $\mathcal O(k^2)$ 的，总复杂度 $\mathcal O(nk^2)$。类似 ZR#309，改为 **在 DFS 序上 DP**。

先用 **后序遍历**（先遍历儿子再遍历根）求出点的 DFS 序。求出链左边的背包：设 $f_{i,j}$ 表示考虑了 DFS 序 $\leq i$ 的点中选了 $j$ 个苹果的最大幸福度。设 $x$ 表示 DFS 序为 $i$ 的点。

- $x$ 的苹果不选，则子树都不能选：$f_{i,j}=f_{i-sz_x,j}$。
- $x$ 的苹果选，则没有限制：$f_{i,j}=f_{i-1,j-k}+kv_x=f_{i-1,j-k}-(j-k)v_x+jv_x\,(k\leq a_x)$，即 $f_{i,j}=\max_{k\in[i-a_x,i]} \{f_{i-1,k}-kv_x\}+jv_x$。可以单调队列优化。

把儿子 reverse 后再求链右边同理（类似前后缀背包合并，链左边是正 DFS 序背包，链右边是逆 DFS 序背包）。

时间复杂度 $\mathcal O(nk)$。

---

upd on 2024.4.20：

设 $f_{x,i}$ 表示按 DFS 序考虑到 $x$，且 $x\leadsto rt$ 的所有节点 $p$ 只考虑了 $a_p-1$ 个苹果，此时选了 $i$ 个苹果的最大幸福度。

$g_{x,i}$ 表示按逆 DFS 序考虑到 $x$，且 $x\leadsto rt$ 的所有节点 $p$ 都没有考虑苹果，此时选了 $i$ 个苹果的最大幸福度。

巧妙实现：

- $f$：DFS 时，先加入 $a_x-1$ 个苹果（不强制至少选一个），等 DFS 完儿子再加入剩下的那个苹果（强制选），那么 $x\leadsto rt$ 上所有点都只算了不强制必须选的 $a_p-1$ 个苹果。

- $g$：DFS 时，等 DFS 完儿子后再加入 $a_x$ 个苹果（不强制至少选一个），那么 $x\leadsto rt$ 上所有点都没有算苹果。

刚 DFS 到 $x$ 时还未考虑 $x$ 的子树，回溯完后已经考虑了 $x$ 的子树。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=2e4+5,K=5e5+5;
int t,n,k,a[N],w[N],sum[N],l,r,q[K],val[K],ans;
vector<int>v[N],f[N],g[N];
void add(int x){
	l=1,r=0;
	for(int i=0;i<=k;i++){
		val[i]=f[x][i]-i*w[x];
		while(l<=r&&val[i]>=val[q[r]]) r--;
		q[++r]=i;
		while(l<=r&&q[l]<i-(a[x]-1)) l++;
		f[x][i]=val[q[l]]+i*w[x];
	}
}
void add2(int x){
	for(int i=k;i>=1;i--) f[x][i]=f[x][i-1]+w[x];
}
void dfs(int x){
	add(x),g[x]=f[x];
	for(int y:v[x]){
		f[y]=f[x],dfs(y);
		for(int i=0;i<=k;i++) f[x][i]=max(f[x][i],f[y][i]);
	}
	add2(x);
}
void dfs2(int x){
	for(int i=0;i<=k;i++)
		ans=max(ans,f[x][i]+g[x][k-i]+sum[x]);
	for(int y:v[x]){
		f[y]=f[x],sum[y]=sum[x]+w[y],dfs2(y);
		for(int i=0;i<=k;i++) f[x][i]=max(f[x][i],f[y][i]);
	}
	add(x),add2(x);
}
signed main(){
	scanf("%d",&t);
	while(t--){
		scanf("%d%d",&n,&k),ans=0;
		for(int i=1;i<=n;i++)
			v[i].clear(),
			vector<int>(k+1).swap(f[i]),vector<int>(k+1).swap(g[i]);	//注意直接 f[i].resize(k+1) 会 MLE
		for(int i=1,x;i<=n;i++)
			scanf("%d%d%d",&x,&a[i],&w[i]),
			v[x].push_back(i);
		dfs(1);
		for(int i=1;i<=n;i++)
			reverse(v[i].begin(),v[i].end()),f[i].assign(k+1,0);
		sum[1]=w[1],dfs2(1);
		printf("%d\n",ans);
	}
	return 0;
}
```

##### C4

P2986 [USACO10MAR] Great Cow Gathering G

换根 DP，第一遍算出 $f_x$ 表示子树的奶牛到 $x$ 的代价，第二遍根据 $f_x$ 自上而下算出 $g_x$ 表示所有奶牛到 $x$ 的代价，设 $sz_x$ 表示 $x$ 子树的奶牛数，$g_y=g_x-sz_y\cdot w_{x,y}+(n-sz_y)\cdot w_{x,y}$。

时间复杂度 $\mathcal O(n)$。

##### C5

P1399 [NOI2013] 快餐店，基环树

> $n$ 个点 $n$ 条边的无向连通图，边有长度。
>
> 需要在图上找一个位置，可以为点也可以在边上，距离点也不必是整数。图上两点间的距离为最短路径长度，使得这个位置与距离最远的点最近。
>
> $1\leq n\leq 10^5$。

点在树直径的中点：离 $x$ 最远的点是树直径的某端，所以取树直径中点最优。

去掉环后求树的直径，现在考虑直径经过了环。对环上每个点求出 $f_x$ 表示 $x$ 到树上最长的距离。

断环成链，求 $\max_{j<i}(f_i+f_j+\min(dis_i-dis_j,all-dis_i+dis_j))$，

固定 $i$，对所有 $j$ 找是 $\min$ 的前半还是后半，形成两个区间（双指针），然后分别求区间 $dis_j$ 的最大值和最小值。

找环一般采用拓扑排序 + dfs。