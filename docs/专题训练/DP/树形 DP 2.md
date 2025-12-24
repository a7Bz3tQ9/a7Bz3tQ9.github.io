依赖背包的另一种写法：

父亲传给儿子，在子树里绕一圈再传回父亲

```cpp
void ins(int *f,int w,int v,int c){
	auto add=[&](int w,int v){
		for(int i=m;i>=w;i--) f[i]=max(f[i],f[i-w]+v);
	};
	int k=0;
	while(c>=(1<<k))
		add(w*(1<<k),v*(1<<k)),c-=1<<k,k++;
	if(c) add(w*c,v*c);
}
void dfs(int x,int fa){
	for(int i=0;i<=m;i++)
		f[x][i]=i>=c[x]?f[fa][i-c[x]]+w[x]:-2e9;
	ins(f[x],c[x],w[x],d[x]-1);
	for(int y:v[x])
		if(y!=fa) dfs(y,x);
	if(fa)
		for(int i=0;i<=m;i++) f[fa][i]=max(f[fa][i],f[x][i]);
}
```

##### P4516 [JSOI2018] 潜入行动（树形 DP + 复杂度分析）

2021.10.26

> 给出一棵 $n$ 个节点的树，有 $k$ 个监听器，可以在节点上放一个监听器或不放，监听器必须用完。如果节点 $x$ 上放了监听器，就能监听所有与 $x$ 直接相邻的节点（注意不能监听 $x$ 本身）。
>
> 求有多少种放监听器的方案，使得所有节点都能被监听。
>
> $1\leq n\leq 10^5$，$1\leq k\leq \min\{n,100\}$。

一眼树形 DP（树上背包）。有两大难点。

难点之一是 **状态的设计**，$f_{i,j,0/1,0/1}$ 表示以 $i$ 为根的子树内放了 $j$ 个监听器，$i$ 是否放了监听器，$i$ 是否被它的儿子监听，在这种情况下的方案数（在以 $x$ 为根的子树中除 $x$ 外的其他节点都被监听到了）。

设计好了状态，转移也就水到渠成了。不过转移上那么多 `01` 写起来比较麻烦，我们可以具体分析一下。

对于这个 `01` 可以用变量表示，假设当前合并的两个背包为 $f_{x,a,p_1,q_1}$ 和 $f_{y,b,p_2,q_2}\,(y\in son_x)$。考虑合并后的 $f_{x,a+b,p_3,q_3}$，$p_3,q_3$ 分别会是什么，以及这个合并什么时候合法。

- $p_3$ 是合并后节点 $x$ 是否放了监听器，这显然不受 $y$ 影响，也就是说 $p_3=p_1$。

- $q_3$ 是合并后节点 $x$ 是否被监听，有两种情况：$x$ 之前已经被监听，$x$ 现在被 $y$ 监听。即：$q_3=q_1\text{ or }p_2$。

但是并不是所有状态都能合并的，可以合并当且仅当 $y$ 被监听。$y$ 被监听同样有两种情况：$y$ 之前被监听，$y$ 现在被 $x$ 监听。也就是说当且仅当 $q_2\text{ or }p_1=1$ 时这两个状态可以合并。

于是得到一个清爽的 4 个 for 转移：

```cpp
for(int p1=0;p1<2;p1++)
	for(int q1=0;q1<2;q1++)
		for(int p2=0;p2<2;p2++)
			for(int q2=0;q2<2;q2++)
				if(q2|p1) (tmp[a+b][p1][q1|p2]+=1ll*f[x][a][p1][q1]*f[y][b][p2][q2]%mod)%=mod;
```

时间复杂度 $\mathcal O(nk)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e5+5,M=110,mod=1e9+7;
int n,k,x,y,sz[N],f[N][M][2][2],tmp[M][2][2];
vector<int>v[N];
void dfs(int x,int fa){ 
	sz[x]=f[x][0][0][0]=f[x][1][1][0]=1;
	for(int y:v[x]){ 
		if(y==fa) continue;
		dfs(y,x);
		for(int i=0;i<=min(sz[x]+sz[y],k);i++) tmp[i][0][0]=tmp[i][0][1]=tmp[i][1][0]=tmp[i][1][1]=0;
		for(int a=0;a<=min(sz[x],k);a++)
			for(int b=0;b<=min(sz[y],k-a);b++) 
				for(int p1=0;p1<2;p1++)
					for(int q1=0;q1<2;q1++)
						for(int p2=0;p2<2;p2++)
							for(int q2=0;q2<2;q2++)
								if(q2|p1) (tmp[a+b][p1][q1|p2]+=1ll*f[x][a][p1][q1]*f[y][b][p2][q2]%mod)%=mod;
		sz[x]+=sz[y];
		for(int i=0;i<=min(sz[x],k);i++)
			for(int p=0;p<2;p++)
				for(int q=0;q<2;q++) f[x][i][p][q]=tmp[i][p][q];
	} 
}
signed main(){
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++){
		scanf("%d%d",&x,&y);
		v[x].push_back(y),v[y].push_back(x);
	}
	dfs(1,0),printf("%d\n",(f[1][k][0][1]+f[1][k][1][1])%mod);
	return 0; 
} 
```

##### P3354 [IOI2005]Riv 河流

2023.3.1 树形 DP

> 给出一棵 $n+1$ 个节点的有根树，编号 $0\sim n$ 且根为 $0$。点有点权 $w_i$，边有边权 $d_i$。
>
> $0$ 为关键点，再从 $1\sim n$ 中选 $k$ 个点作为关键点，代价为 每个点的点权 $\times$ 到离它最近的关键点祖先（包括它自己）的距离 的总和。求最小代价。
>
> $2\leq n\leq 100$，$1\leq k\leq \min(n,50)$，$0\leq w_i\leq 10^4$，$0\leq d_i\leq 10^4$。

设 $f_{x,i,j,0/1}$ 表示子树 $x$ 选择了 $i$ 个关键点，钦定 $x$ 及其祖先中离 $x$ 最近的关键点是 $j$，$x$ 是否是关键点，此时的最小代价。转移见代码。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=110;
int n,k,x,z,w[N],f[N][N][N/2][2],tmp[N][N/2][2],d[N];
vector<pair<int,int> >v[N];
vector<int>anc;
void dfs(int x){
	anc.push_back(x);
	for(int i:anc) f[x][i][1][1]=f[x][i][0][0]=0;
	for(auto p:v[x]){
		int y=p.first;
		d[y]=d[x]+p.second,dfs(y);
		for(int i:anc)
			for(int j=0;j<=k;j++)
				for(int o=0;o<2;o++) tmp[i][j][o]=f[x][i][j][o],f[x][i][j][o]=1e9;
		for(int i:anc)
			for(int a=0;a<=k;a++)
				for(int b=0;b<=k-a;b++)
					for(int o=0;o<2;o++){
						int j=o?x:i;
						f[x][i][a+b][o]=min(f[x][i][a+b][o],tmp[i][a][o]+min(f[y][j][b][0],f[y][j][b][1]));
					}
	}
	for(int i:anc)
		for(int j=0;j<=k;j++) f[x][i][j][0]+=(d[x]-d[i])*w[x];
	anc.pop_back();
}
signed main(){
	scanf("%d%d",&n,&k),k++;	//根据题意一共选 k+1 个关键点且根必选
	for(int i=2;i<=n+1;i++)
		scanf("%d%d%d",&w[i],&x,&z),v[x+1].push_back({i,z});
	memset(f,0x3f,sizeof(f));
	dfs(1),printf("%d\n",f[1][1][k][1]);
	return 0;
}
```

##### P4099 [HEOI2013]SAO

> 有一个有向图 $G$，其基图是一棵树。求它拓扑序的个数 $\bmod 10^9+7$。
>
> $T\leq 5$，$1\leq n\leq 1000$。

求“拓扑排列个数”是 NP 问题，只能用指数级的状压 $dp$ 一类算法解决。而 $n\leq 1000$，暗示我们要充分利用“$G$ 的基图是一棵树”这个条件。

树形 DP，设 $f_{x,i}$ 表示将以 $x$ 为根的子树拓扑排序后，$x$ 位于拓扑序的第 $i$ 位的方案数。

转移 $f_{x,i},f_{y,j}\to new\ f_{x,k}$，新序列中，比 $x$ 小的 $k-1$ 个有 $i-1$ 个来自原序列，方案数 $\large\binom{k-1}{i-1}$，同理，填好比 $x$ 大的方案数为 $\large\binom{sz_x+sz_y-k}{sz_x-i}$。故 $new\ f_{x,k}\gets f_{x,i}\times f_{y,j}\times \large\binom{k-1}{i-1}\large\binom{sz_x+sz_y-k}{sz_x-i}$。

- 如果要求 $x$ 的拓扑序小于 $y$ 的拓扑序：

  注意到，$i,j$ 确定后，$k$ 取到的范围是一个区间。因为 $x$ 序列位置的相对顺序不变，而 $x$ 的拓扑序更小，故原本 $y$ 后面的数现在也在 $x$ 后面，$y$ 前面的 $j-1$ 个数可以在 $x$ 前也可以在 $x$ 后，故 $k\in[(i-1)+1,(i-1)+(j-1)+1]=[i,i+j-1]$。

  直接转移 $\mathcal O(n^3)$。注意到后面的组合数与 $j$ 无关，所以使用前缀和优化：枚举 $i,k$，算出 $j$ 的范围即可。

  ```cpp
  for(int i=1;i<=sz[x];i++)
  	for(int j=1;j<=sz[y];j++)
  		for(int k=i;k<=i+j-1;k++) transfer;
  //改变循环顺序：
  for(int i=1;i<=sz[x];i++)
  	for(int k=i;k<=i+sz[y]-1;k++)
  		for(int j=k-i+1;j<=sz[y];j++) transfer; 
  ```

- 如果要求 $x$ 的拓扑序大于 $y$ 的拓扑序：

  类似，$k\in [(i-1)+j+1,(i-1)+j+(sz_y-j)+1]=[i+j,i+sz_y]$。

  ```cpp
  for(int i=1;i<=sz[x];i++)
  	for(int j=1;j<=sz[y];j++)
  		for(int k=i+j;k<=i+sz[y];k++) transfer;
  //改变循环顺序：
  for(int i=1;i<=sz[x];i++)
  	for(int k=i+1;k<=i+sz[y];k++)
  		for(int j=1;j<=k-i;j++) transfer; 
  ```

时间复杂度 $\mathcal O(n^2)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e3+5,mod=1e9+7;
int t,n,x,y,z,sz[N],f[N][N],s[N][N],c[N][N],tmp[N];
vector<pair<int,int> >v[N];
void dfs(int x,int fa){
	f[x][1]=sz[x]=1;
	for(auto p:v[x]){
		int y=p.first;
		if(y==fa) continue; 
		dfs(y,x);
		for(int i=1;i<=sz[x];i++) tmp[i]=f[x][i],f[x][i]=0;
		if(p.second==1){
			for(int i=1;i<=sz[x];i++)
				for(int k=i;k<=i+sz[y]-1;k++)
					f[x][k]=(f[x][k]+1ll*tmp[i]*(s[y][sz[y]]-s[y][k-i]+mod)%mod*c[k-1][i-1]%mod*c[sz[x]+sz[y]-k][sz[x]-i]%mod)%mod;
		} 
		else{
			for(int i=1;i<=sz[x];i++)
				for(int k=i+1;k<=i+sz[y];k++)
					f[x][k]=(f[x][k]+1ll*tmp[i]*s[y][k-i]%mod*c[k-1][i-1]%mod*c[sz[x]+sz[y]-k][sz[x]-i]%mod)%mod; 
		} 
		sz[x]+=sz[y]; 
	}
	for(int i=1;i<=sz[x];i++) s[x][i]=(s[x][i-1]+f[x][i])%mod;
}
signed main(){
	scanf("%d",&t),c[0][0]=1;
	for(int i=1;i<=1e3;i++){
		c[i][0]=1;
		for(int j=1;j<=i;j++) c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	}
	while(t--){
		scanf("%d",&n);
		for(int i=1;i<=n;i++) v[i].clear();
		for(int i=1;i<n;i++){
			char c;
			scanf("%d %c%d",&x,&c,&y),x++,y++,z=(c=='>'?1:-1);
			v[x].push_back({y,z}),v[y].push_back({x,-z});
		}
		memset(f,0,sizeof(f));
		dfs(1,0),printf("%d\n",s[1][n]);
	}
	return 0;
}
```

##### 树拓扑序

> 给出一棵 $n$ 个节点的外向树，求它所有合法拓扑序的逆序对个数之和 $\bmod 10^9+7$。
>
> 一个拓扑序 $p$ 是合法的，当且仅当 $p$ 是一个 $1\sim n$ 的排列，且对于所有有向边 $x\to y$，$p$ 中都有 $y$ 在 $x$ 前面出现。
>
> 注意：本题中拓扑序的定义和一般的定义略有不同。
>
> $1\leq n\leq 500$。

设 $f_x$ 表示 $x$ 子树的拓扑序数量，$g_x$ 表示 $x$ 子树中所有拓扑序的逆序对数之和。
$$
f_x'=f_xf_y\times \binom{sz_x+sz_y-1}{sz_y}\\
g_x'=(g_xf_y+g_yf_x)\binom{sz_x+sz_y-1}{sz_y}+(x,y\ 之间产生的逆序对数)
$$
（弯路：不知道归并两个子树怎么算 $x,y$ 之间产生的逆序对数。其实要拆贡献的提示已经很明显了，既然直接算不了那就拆拆拆）

设 $dp_{x,pos,w}$ 表示子树 $x$ 的所有拓扑序中，把节点 $w$ 放在第 $pos$ 个位置的方案数（其他点在满足拓扑序的条件下任意放）。

枚举子树 $y$ 的 $pos,w$，枚举合并完序列后 $w$ 前面有 $c$ 个 $x$ 的点：
$$
dp_{x,pos+c,w}'\gets f_x\times dp_{y,pos,w} \binom{c+pos-1}{c}\binom{sz_{x}-c+sz_{y}-pos-1}{sz_y-pos}\\
g_x'\gets (\sum_{i\leq c,j>w}dp_{x,i,j}+\sum_{i>c,j<w}dp_{x,i,j})\times dp_{y,pos,w}\binom{c+pos-1}{c}\binom{sz_{x}-c+sz_{y}-pos-1}{sz_y-pos}\\
$$
（算 $(\sum_{i\leq c,j>w}+\sum_{i>c,j})$ 可以二维前缀和）

$w\in subtree_x$ 的 $dp_{x,*,w}'$ 还没算。枚举子树 $x$ 的 $pos,w$，枚举合并完后 $w$ 前面有 $c$ 个 $y$ 的点：
$$
dp_{x,pos+c,w}'\gets f_y\times dp_{x,pos,w}\binom{c+pos-1}{c}\binom{sz_{x}-pos+sz_{y}-c-1}{sz_{y}-c}
$$
时间复杂度 $\mathcal O(n^3)$，因为枚举 $pos,c$ 的总复杂度是 $\mathcal O(n^2)$ 的，枚举 $w$ 还有 $\mathcal O(n)$ 的复杂度。





注意：本题中拓扑序的定义和一般的定义略有不同，以下所说“拓扑序”均是按照本题中的定义。

考虑DP。设$g[u]$为以$u$为根的子树的拓扑序数量。这个比较便于计算：在把一个儿子$v$的子树合并进来时令$g[u]=g[u]g[v]{sz[v]+sz[u]-1\choose sz[v]}$即可（注：这里$sz[u]$尚没有加上$sz[v]$，在全部转移完成后才会加上，下同）。

考虑如果求逆序对数和。设$f[u]$表示以$u$为根的子树的所有拓扑序下的逆序对数之和。在把$u$和一个儿子$v$合并时，我们分三个部分来计算$f[u]$。为了方便表示，不妨设合并前的$f[u]$为$f'[u]$，$g[u]$为$g'[u]$。

1. $u$除$v$子树外的其他部分，这些节点内部的逆序对数。对新$f[u]$的贡献为：$f'[u]\cdot g[v]\cdot{sz[v]+sz[u]-1\choose sz[v]}$。
2. $v$的子树内部的逆序对数。对新$f[u]$的贡献为$f[v]\cdot g'[u]\cdot {sz[v]+sz[u]-1\choose sz[v]}$。
3. $u,v$两部分节点之间产生的逆序对数。

第三部分就比较难以计算了。但也是有方法的。

我们设$dp[u][pos][w]$表示在以$u$为根的子树的所有拓扑序中，把$w$这个节点放在第$pos$个位置（其他点任意放，但要满足拓扑序）时的方案数。

转移（合并$u,v$）时，我们枚举$v$子树中的一个节点$w$，再枚举它在$v$子树的拓扑序里的位置$pos$，再枚举$u$的子树中有$c$个节点在合并后的拓扑序中排在$w$前面。则此时产生的逆序对数位$\sum_{i\leq c,j>w}dp[u][i][j]+\sum_{i>c,j<w}dp[u][i][j]$，这相当于二维平面上右上角和左下角两个区域的和，可以用二维前缀和预处理后$O(1)$求出。把这个数量累加到新的$dp[u][pos+c][w]$中。

用这个数量再乘以方案数$dp[v][pos][w]\cdot g'[u]\cdot{c+pos-1\choose c}\cdot{sz[u]-c-1+sz[v]-pos\choose sz[v]-pos}$，就是对新的$f[u]$的贡献了。其中两个组合数分别表示把前半部分和后半部分任意归并起来的方案数。

这是枚举了$v$的子树中的一个节点。我们再枚举$u$子树中的一个节点，考虑它对新的$dp[u][\dots][\dots]$的贡献。方法是类似的。

众所周知在树形DP时，枚举$sz[u]$再枚举$sz[v]$的总复杂度是$O(n^2)$的。故本题的总复杂度是$O(n^3)$的。

##### P6992 [NEERC2014]Hidden Maze

2022.9.23 中位数 + 树形 DP，一道有启发性的题

> 给出一棵 $n$ 个节点的树，边有边权。对于无序点对 $(x,y)$，若 $x,y$ 的路径有奇数条边，定义 $f(x,y)$ 为路径 $x\leadsto y$ 所有边边权的中位数。
>
> 设 $S$ 为所有距离为奇数的无序对构成的集合，求 $\frac{1}{|S|}\sum_{(x,y)\in S}f(x,y)$。
>
> $2\leq n\leq 3\times 10^4$，$1\leq c_i\leq 10^6$，保证树以某种随机方式生成。

设 $D_c$ 表示中位数为 $c$ 的路径集合，$ans=\large\frac{\sum_c c\cdot |D_c|}{\sum_c|D_c|}$，只要求 $|D_c|$ 即可。

按边权从小到大枚举所有边，设当前边边权为 $0$、前面的边权为 $1$、后面的边权为 $-1$，那么经过这条边的边权和为 $0$ 的路径的中位数就是这条边。

断掉 $(x,y)$，分裂出两棵子树 $T_x,T_y$。设 $f_{x,i}$ 表示以 $x$ 为根的子树中有多少点到 $x$ 的边权和为 $i$，修改时暴力修改祖先。假设 $T_x$ 是“正常的”子树，另一棵是子树的补，前者直接用 $f$，后者可以枚举 $y$ 的祖先作为路径两端的 LCA，类似点分治减掉两端在同个子树的方案数。

由于树随机生成，期望树高 $\mathcal O(\sqrt n)$，时间复杂度期望 $\mathcal O(n\sqrt n)$。

```cpp
//O2
#include<bits/stdc++.h>
using namespace std;
const int N=3e4+5;
int n,x,y,z,s[N],top,o=300,fa[N],w[N],mxd[N],f[N][610];
long long sz,ans,cnt;
struct edge{int x,y,z;}e[N];
vector<int>v[N]; 
void upd(int x,int v){
	for(int i=-mxd[x];i<=mxd[x];i++) f[fa[x]][i+w[x]+o]+=f[x][i+o]*v;
}
void dfs(int x){
	f[x][o]=1;
	for(int y:v[x]) if(y!=fa[x])
		fa[y]=x,dfs(y),mxd[x]=max(mxd[x],mxd[y]+1),w[y]=1,upd(y,1);
}
signed main(){
	scanf("%d",&n);
	for(int i=1;i<n;i++){
		scanf("%d%d%d",&x,&y,&z),e[i]={x,y,z};
		v[x].push_back(y),v[y].push_back(x);
	}
	sort(e+1,e+n,[](edge x,edge y){return x.z<y.z;}),dfs(1);
	for(int i=1;i<n;i++){
		int x=e[i].x,y=e[i].y,z=e[i].z;
		if(fa[y]==x) swap(x,y);
		sz=top=0;
		for(int j=x;j;j=fa[j]) s[++top]=j;
		for(int j=top-1;j>=1;j--) upd(s[j],-1);
		for(int j=2,len=0;j<=top;j++){
			for(int k=-mxd[x];k<=mxd[x];k++) sz+=1ll*f[x][k+o]*f[s[j]][-k-len+o];
			len+=w[s[j]];
		}
		ans+=sz*z,cnt+=sz,w[x]=-1;
		for(int j=1;j<top;j++) upd(s[j],1);
	}
	printf("%.9lf\n",1.0*ans/cnt);
	return 0;
}
```

##### 小练习

> 给出一棵 $n$ 个点的树和一个点集 $S$。需要用 $m$ 种颜色对树染色。
>
> 定义一个染色方案是合法的，当且仅当所有相邻的点的颜色都不相同。
>
> 对于所有合法的染色方案 $c_{1\sim n}$，记 $S_c=\{c_x\mid x\in S\}$，求 $|S_c|$ 之和 $\bmod 998244353$。
>
> $1\leq n,m\leq 10^5$，$1\leq |S|\leq n$。

$ans=\sum_{c_{1\sim n}}|S_c|=\sum_{c_{1\sim n}}\sum_{i=1}^m[i\in S_c]=\sum_{i=1}^m\sum_{c_{1\sim n}}[i\in S_c]$。即对于每个颜色 $i$，计算有多少种合法的染色方案 使得 $i\in S_c$。本质是贡献法。

补集转化成，计算有多少种合法的染色方案使得 $i\notin S_c$。

设 $f_{x,0/1}$ 表示若 $x$ 的颜色已经定好，且是否是 $i$ 的情况下，$x$ 子树内合法的染色方案数。则 $f_{x,0}=\prod_{y\in son_x}(f_{y,0}(m-2)+f_{y,1})$，$f_{x,1}=[x\notin S]\prod_{y\in son_x}f_{y,0}(m-1)$。注意 $x$ 的颜色是 $fa_x$ 处定的。

（可能可以这么想到：设 $f_{x,a}$ 表示 $c_x=a$，$x$ 子树内合法的染色方案数。则 $f_{x,a}=\prod_{y\in son_x}(\sum_{b\neq a}f_{y,b})$。然后发现除了 $i$ 之外的颜色地位都是平等的，可以改为设 $f_{x,0/1}$）

- 本题树形 DP 的一个重要思想：自上而下推（用父亲颜色定儿子颜色），而不是自下而上合并（用儿子颜色定父亲颜色）。因为如果自下而上合并，通过 $x$ 儿子染了有几种不同的颜色 来算 $x$ 能染多少颜色，不太好做。但若从上往下钦定，根据 $x$ 的颜色容易算出 $x$ 每个儿子能染多少颜色。

$ans=m (f_{1,0}(m-1)+f_{1,1})$（显然对不同的 $i$ 求出的 $f$ 都是一样的）。

时间复杂度 $\mathcal O(n)$。