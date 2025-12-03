#### 一、wqs 二分

> 用 wqs 二分消掉 $k$ 的限制。

考虑一个函数 $f$，求 $f(k)\,(k\in\mathbb Z)$。如果直接求点值很麻烦，但 $f$ 满足凸性且顶点好算，我们就可以拉伸凸包使 $k$ 成为顶点来找 $f(k)$。二分就是取找拉伸的度。

怎么拉伸？凸函数 $f(x)$ 加上一个一次函数得到的 $g(x)=f(x)+kx$ 也是凸的，且随着斜率的调整，$g(x)$ 的顶点也会不断变化。例如一个上凸壳，$k$ 越大顶点越靠右。

如果能 $\mathcal O(n)$ 求出顶点，就能 $\mathcal O(n\log V)$ 解决问题。

对于形如“恰好选 $k$ 个”的最优化问题，设 $f_i$ 表示恰好选 $i$ 个时的答案，如果 $f$ 是凸的且“选若干个最优”好算，就可以 wqs 二分。

##### 分 k 段平方和

> 给出一个正整数序列 $a$，将其分为 $k$ 段。一段的代价是和的平方，求最小代价。

设 $F(i)$ 表示恰好分 $i$ 段的最小代价。猜测 $F$ 是下凸的。顶点：分成若干段使代价最小。

$G(i)\gets F(i)+c\cdot i$，当 $c=0$ 时显然最优的 $i$ 为 $n$，每多分一段就 $+c$ 的代价，那么随着 $c$ 的增大，最优的 $i$ 会越来越小。求最优的 $i$：设 $f_i$ 表示前 $i$ 个数划分为若干段的最小代价，以及此时划分了多少段，可以斜率优化。

这实际上就是在不断调整将顶点的横坐标挪到 $k$ 的过程，根据顶点 $(k,g_k)$ 算出 $f_k$。

##### 细节：凸壳上的点共线

由于不一定是严格的凸函数，我们算顶点的时候，有可能出现 $g_l=\cdots=g_r$ 的情况，根据我们求解顶点的方式（DP / 贪心），只能算出顶点横坐标是 $l$ 或 $r$，算不出中间的 $k$（比如：调大了算出 $r$，调小了算出 $l$，永远无法算出 $k$）。也就是没办法将顶点横坐标挪到 $k$ 了。

<img src="https://img2022.cnblogs.com/blog/1859218/202207/1859218-20220704121528638-366968041.png" alt="image" style="zoom: 67%;" />

解决方法：例如求的是 $r$，用 $g_r-mid\cdot k$ 更新答案。

#### 二、例题

wqs 二分去掉 $k$ 的限制后，一般和 DP / 贪心结合。注意大前提：$F$ 是凸的。

##### 1. P2619 [国家集训队]Tree I

2022.7.4

> 给出一个 $n$ 个点 $m$ 条边的无向带权连通图，边是黑色或白色。求恰好有 $need$ 条白边的 MST。保证有解。
>
> $n\leq 5\times 10^4$，$m\leq 10^5$，边权 $\in[1,100]$。

设 $f_i$ 表示恰好选 $i$ 条白边的 MST。$f_i+c\cdot i$ 相当于给每条白边加权 $c$，MST 中白边的数量随着 $c$ 的增大而减小。当某一时刻 MST 中白边数量为 $t$ 时，原图的 MST 为经处理后的 MST - $c\cdot t$。

注意有可能出现白边和黑边边权相同的情况。排序时若边权相同白边优先，相当于 $t$ 最接近 $need$ 时是我们上面所说细节的 $r$。故当 $t\geq need$ 时用 MST - $mid\cdot need$ 更新答案。

时间复杂度 $\mathcal O(m\log n\log w)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e5+5;
int n,m,k,f[N],ans;
struct edge{int x,y,z,c;}a[N];
int find(int x){return x==f[x]?x:f[x]=find(f[x]);}
bool ok(int v){
	for(int i=1;i<=n;i++) f[i]=i;
	sort(a+1,a+1+m,[](edge x,edge y){return x.z!=y.z?x.z<y.z:x.c<y.c;});
	int t=0,sum=0;
	for(int i=1;i<=m;i++){
		int x=find(a[i].x),y=find(a[i].y);
		if(x!=y) t+=!a[i].c,f[y]=x,sum+=a[i].z;
	}
	if(t>=k) ans=sum-k*v;	//注意是 >= 不是 ==
	return t>=k;
}
signed main(){
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=m;i++)
		scanf("%d%d%d%d",&a[i].x,&a[i].y,&a[i].z,&a[i].c),a[i].x++,a[i].y++;
	int l=-100,r=100;
	while(l<=r){
		int mid=(l+r)/2;
		for(int i=1;i<=m;i++) a[i].z+=(!a[i].c)*mid;
		if(ok(mid)) l=mid+1;
		else r=mid-1;
		for(int i=1;i<=m;i++) a[i].z-=(!a[i].c)*mid;
	}
	printf("%d\n",ans);
	return 0;
}
```

##### 2. CF125E MST Company（\*2400）

2022.7.4 同 [P5633 最小度限制生成树](https://www.luogu.com.cn/problem/P5633)

> 给出一张 $n$ 个点 $m$ 条边的无向图，求恰好有 $k$ 条从 $1$ 连向其余点的边的 MST，要求输出方案。无解输出 $-1$。
>
> $1\leq n\leq 5000$，$0\leq m\leq 10^5$，$0\leq k\leq 5000$，$1\leq w\leq 10^5$。

记与 $1$ 相连的边为特殊边。设 $f_i$ 表示恰好选 $i$ 条特殊边的 MST。

考虑 $f_1$ 一定是 $2\sim n$ 的 MST 加上与 $1$ 相连权值最小的边，$f_i$ 一定是 $f_{i-1}$ 先加上一条与 $1$ 相连权值最小的边 $e$，再去掉环上与 $1$ 相连权值最大的边，每次的 $e$ 一定不比上一次优，所以 $(i,f_i)$ 是上凸的。

二分 $c$，$g_i\gets f_i+c\cdot i$。给边排序时若边权相同特殊边优先，也就是算出细节中的 $r$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e5+5;
int n,m,k,x,y,z,t,f[N],cnt,vis[N],ans=1e9;
struct edge{int x,y,z,id;}a[N];
bool cmp(edge x,edge y){return x.z!=y.z?x.z<y.z:x.x<y.x;}
int find(int x){return x==f[x]?x:f[x]=find(f[x]);}
bool ok(int c){
	for(int i=1;i<=m;i++) a[i].z+=(a[i].x==1)*c;
	t=0,sort(a+1,a+1+m,cmp);
	for(int i=1;i<=n;i++) f[i]=i;
	for(int i=1;i<=m;i++){
		x=find(a[i].x),y=find(a[i].y);
		if(x!=y) f[x]=y,t+=a[i].x==1;
	}
	for(int i=1;i<=m;i++) a[i].z-=(a[i].x==1)*c;
	return t>=k;
}
void getans(int c){
	for(int i=1;i<=m;i++) a[i].z+=(a[i].x==1)*c;
	cnt=t=0,sort(a+1,a+1+m,cmp);
	for(int i=1;i<=n;i++) f[i]=i;
	for(int i=1;i<=m;i++){
		x=find(a[i].x),y=find(a[i].y);
		if(x!=y&&t+(a[i].x==1)<=k) f[x]=y,cnt++,t+=a[i].x==1,vis[i]=1;
	}
	if(cnt<n-1||t<k) puts("-1"),exit(0);
	printf("%d\n",n-1);
	for(int i=1;i<=m;i++) if(vis[i]) printf("%d ",a[i].id);
	puts("");
} 
signed main(){
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=m;i++)
		scanf("%d%d%d",&x,&y,&z),a[i]={min(x,y),max(x,y),z,i};
	int l=-1e5,r=1e5;
	while(l<=r){
		int mid=(l+r)/2;
		if(ok(mid)) ans=mid,l=mid+1;
		else r=mid-1;
	}
	if(ans==1e9) puts("-1"),exit(0);
	getans(ans);
	return 0; 
}
```

P5633 最小度限制生成树：不输出方案。

```cpp
//O2
#include<bits/stdc++.h>
using namespace std;
const int N=5e5+5;
int n,m,s,k,x,y,z,t,f[N],cnt;
long long sum,ans=1e9;
struct edge{int x,y,z,f;}a[N];
bool cmp(edge x,edge y){return x.z!=y.z?x.z<y.z:x.f>y.f;}
int find(int x){return x==f[x]?x:f[x]=find(f[x]);}
bool ok(int c){
	for(int i=1;i<=m;i++) a[i].z+=a[i].f*c;
	t=sum=0,sort(a+1,a+1+m,cmp);
	for(int i=1;i<=n;i++) f[i]=i;
	for(int i=1;i<=m;i++){
		x=find(a[i].x),y=find(a[i].y);
		if(x!=y) f[x]=y,t+=a[i].f,sum+=a[i].z;
	}
	for(int i=1;i<=m;i++) a[i].z-=a[i].f*c;
	if(t>=k) ans=sum-1ll*c*k;
	return t>=k;
}
signed main(){
	scanf("%d%d%d%d",&n,&m,&s,&k);
	for(int i=1;i<=n;i++) f[i]=i;
	for(int i=1;i<=m;i++){
		scanf("%d%d%d",&x,&y,&z),a[i]={x,y,z,x==s||y==s};
		if(!a[i].f) f[find(x)]=find(y),cnt++;
	}
	if(cnt+k<n-1) puts("Impossible"),exit(0);
	int l=-3e4,r=3e4;
	while(l<=r){
		int mid=(l+r)/2;
		if(ok(mid)) l=mid+1;
		else r=mid-1;
	}
	if(ans==1e9) puts("Impossible");
	else printf("%lld\n",ans);
	return 0; 
}
```

##### 3. CF739E Gosha is hunting（\*3000）

2022.7.4

> 有 $n$ 只 Pokemons，$a$ 个 A 类球和 $b$ 个 B 类球。A 类球、B 类球抓到第 $i$ 只 Pokemons 的概率分别为 $p_i,u_i$。对于同一只 Pokemons，同类的球只能使用 $\leq 1$ 个，但是可以同时使用 A 类球和 B 类球（都抓到算一个）。
>
> 合理分配每个球抓哪只 Pokemons，使得抓到 Pokemons 的总只数期望最大。输出这个值。
>
> $2\leq n\leq 2000$，$0\leq a,b\leq n$，$0\leq p_i,u_i\leq 1$。

$n^3$ dp：设 $f_{i,j,k}$ 表示考虑了前 $i$ 只 Pokemons，用了 $j$ 个 A 类球和 $k$ 个 B 类球的期望最大只数。$f_{i,j,k}=\max\{f_{i-1,j,k},f_{i-1,j-1,k}+p_i,f_{i-1,j,k-1}+q_i,f_{i-1,j-1,k-1}+p_i+u_i-p_iu_i\}$，$ans=f_{n,a,b}$。

考虑对 A 类球 wqs 二分，$F(j)$ 表示用了恰好 $j$ 个 A 类球时的 $f_{n,j,b}$。顶点：用多少 A 类球时最大，$\mathcal O(nb)$ DP 即可。实现时可以 (用了多少 A 类球，期望最大只数) 二元组重载 < 和 +。然后套路地 $G(j)\gets F(j)-c\cdot j$，即每用一个 A 类球需要 $c$ 的代价。

重载 < 时期望相同时，A 多的算小。也就是尽可能让答案相同时 A 用的少，算出“细节”中说的 $l$，当 A 数量 $\leq a$ 时用 $f_{n,b}+c\cdot mid$ 更新答案。

时间复杂度 $\mathcal O(nb\log a)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=2e3+5;
int n,x,y;
double a[N],b[N],eps=1e-8,ans;
struct dp{
	int x; double ans;
	bool operator<(dp a)const{return fabs(ans-a.ans)>eps?ans<a.ans:x>a.x;}	//要写 const
	dp operator+(dp a){return {x+a.x,ans+a.ans};}
}f[N][N],mx; 
signed main(){
	scanf("%d%d%d",&n,&x,&y);
	for(int i=1;i<=n;i++) scanf("%lf",&a[i]);
	for(int i=1;i<=n;i++) scanf("%lf",&b[i]);
	double l=0,r=1;
	while(l+eps<r){
		double mid=(l+r)/2;
		for(int i=1;i<=n;i++)
			for(int j=0;j<=y;j++){
				f[i][j]=f[i-1][j];
				f[i][j]=max(f[i][j],f[i-1][j]+(dp){1,a[i]-mid});
				if(j){
					f[i][j]=max(f[i][j],f[i-1][j-1]+(dp){0,b[i]});
					f[i][j]=max(f[i][j],f[i-1][j-1]+(dp){1,a[i]+b[i]-a[i]*b[i]-mid});
				}
			}
		if(f[n][y].x<=x) ans=f[n][y].ans+x*mid,r=mid;
		else l=mid;
	}
	printf("%.10lf\n",ans);
	return 0;
}
```

还可以 wqs 二分套 wqs 二分，时间复杂度 $\mathcal O(n\log a\log b)$：

upd on 2023.4.29：wqs 二分套 wqs 二分的做法是 [假](https://www.luogu.com.cn/discuss/422583) 的。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=2e3+5;
int n,x,y;
double a[N],b[N],eps=1e-8,ans;
struct dp{
	int x,y; double ans;
	bool operator<(dp a)const{return fabs(ans-a.ans)>eps?ans<a.ans:(y!=a.y?y>a.y:x>a.x);}	//先排 y 再排 x
	dp operator+(dp a){return {x+a.x,y+a.y,ans+a.ans};}
}f[N],mx; 
void calc(double ca,double cb){
	for(int i=1;i<=n;i++)
		f[i]=max({f[i-1],f[i-1]+(dp){1,0,a[i]-ca},f[i-1]+(dp){0,1,b[i]-cb},f[i-1]+(dp){1,1,a[i]+b[i]-a[i]*b[i]-ca-cb}});
}
signed main(){
	scanf("%d%d%d",&n,&x,&y);
	for(int i=1;i<=n;i++) scanf("%lf",&a[i]);
	for(int i=1;i<=n;i++) scanf("%lf",&b[i]);
	double l=0,r=1;
	while(l+eps<r){
		double m1=(l+r)/2,L=0,R=1,p=-1,m2;
		while(L+eps<R){
			calc(m1,m2=(L+R)/2);
			if(f[n].y<=y) p=R=m2;
			else L=m2;
		} 
		calc(m1,p);
		if(f[n].x<=x) ans=f[n].ans+x*m1+y*p,r=m1;
		else l=m1;
	}
	printf("%.10lf\n",ans);
	return 0;
}
```

此外，还可以网络流：新建两个点 $A,B$，连 $(S,A,a,0)$，$(S,B,b,0)$，对于每个球 $i$，连 $(A,i,1,p_i)$，$(B,i,1,u_i)$，$(i,T,1,0)$，$(i,T,1,-u_ip_i)$（表示如果两类球都选了要 $-u_ip_i$），然后跑最大费用最大流即可。

##### 4. P4383 [八省联考 2018] 林克卡特树

2022.7.4

> 给出一棵 $n$ 个节点的树，边有边权，要求删除其中恰好 $k$ 条边，再加入 $k$ 条 $0$ 权边使得其仍是一棵树，最大化树的直径。
>
> $1\leq n\leq 3\times 10^5$，$0\leq k\leq \min(3\times 10^5,n-1)$，$|w|\leq 10^6$。

删除 $k$ 条边后，原树被分为 $k+1$ 个连通块，用 $0$ 权边连接后的最大直径，实际上就是 $k+1$ 个连通块直径的和。相当于要选 $k+1$ 条不相交的链（对应连通块中的直径），使得这些链的边权和最大。

设 $F(i)$ 表示恰好选了 $i$ 条链的最大边权和。$G(i)\gets F(i)+c\cdot i$ 后，$c$ 越大，最优的 $i$ 越大。二分 $c$。

- 设 $f_{x,i}\,(i\in[0,2])$ 表示 $x$ 的子树中，$x$ 下面接了 $i$ 条边（分别对应不在链上/作为链的端点/链的中间点）的最大边权和，**且 $i=1$ 时与 $x$ 相连的链不计入链的总数**。这是记录路径延伸情况的常见套路。

- 再设 $g_x=\max(f_{x,0},f_{x,1}+c,f_{x,2})$，即 $i=1$ 时链在 $x$ 结束了，此时计入链的总数。$ans=g_1$。

$$
\begin{aligned}
f_{x,0}&=\max(f_{x,0},f_{x,0}+g_y)\\
f_{x,1}&=\max(f_{x,1},f_{x,0}+f_{y,1}+w_{x,y},f_{x,1}+g_y)\\
f_{x,2}&=\max(f_{x,2},f_{x,1}+f_{y,1}+w_{x,y}+c,f_{x,2}+g_y)
\end{aligned}
$$

注意，要按 $f_{x,2},f_{x,1},f_{x,0}$ 的顺序转移。注意 $c$ 有可能为负，二分下界不能设 $0$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=3e5+5;
int n,k,x,y,z,c;
ll ans;
vector<pair<int,int> >v[N];
struct dp{
	int x; ll f;
	dp operator+(dp a){return {x+a.x,f+a.f};}
	bool operator<(dp a)const{return f!=a.f?f<a.f:x>a.x;}	//细节中说的 l，是 > 不是 <，因为 x 小的算大，反之 x 大的算小
}f[N][3],g[N];
void dfs(int x,int fa){
	f[x][0]=f[x][1]={0,0},f[x][2]={0,(ll)-1e9};
	for(auto p:v[x]){
		int y=p.first,z=p.second;
		if(y!=fa){
			dfs(y,x);
			f[x][2]=max({f[x][2],f[x][1]+f[y][1]+(dp){1,z+c},f[x][2]+g[y]});
			f[x][1]=max({f[x][1],f[x][0]+f[y][1]+(dp){0,z},f[x][1]+g[y]});
			f[x][0]=max({f[x][0],f[x][0]+g[y]});
		}
	}
	g[x]=max({f[x][0],f[x][1]+(dp){1,c},f[x][2]});
}
signed main(){
	scanf("%d%d",&n,&k),k++;
	for(int i=1;i<n;i++){
		scanf("%d%d%d",&x,&y,&z);
		v[x].push_back({y,z}),v[y].push_back({x,z});
	}
	dfs(1,0);
	int l=-1e8,r=1e8;
	while(l<=r){
		int mid=(l+r)/2;
		c=mid,dfs(1,0);
		if(g[1].x<=k) ans=g[1].f-1ll*k*c,l=mid+1;
		else r=mid-1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

upd：可以直接设 $f_{x,i}$ 表示 $x$ 子树内，选择若干路径，其中 $x$ 相连的边选了 $i$ 条的最大边权和，以及此时选择的路径最多是多少条。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=3e5+5;
int n,k,x,y,z,c;
ll ans;
vector<pair<int,int> >v[N];
struct dp{
	int x; ll f;
	dp operator+(dp a){return {x+a.x,f+a.f};}
	bool operator<(dp a)const{return f!=a.f?f<a.f:x>a.x;}
}f[N][3],g[N];
void dfs(int x,int fa){
	f[x][0]={0,0},f[x][1]={0,(ll)-1e9},f[x][2]={1,c};	//由于可以一个点作为一条链，所以 f[x][2]={1,c} 
	for(auto p:v[x]){
		int y=p.first,z=p.second;
		if(y!=fa){
			dfs(y,x);
			f[x][2]=max({f[x][2],f[x][2]+g[y],f[x][1]+f[y][0]+(dp){0,z},f[x][1]+f[y][1]+(dp){-1,z-c}});
			f[x][1]=max({f[x][1],f[x][1]+g[y],f[x][0]+f[y][1]+(dp){0,z},f[x][0]+f[y][0]+(dp){1,z+c}});
			f[x][0]=max({f[x][0],f[x][0]+g[y]});
		}
	}
	g[x]=max({f[x][0],f[x][1],f[x][2]});
}
signed main(){
	scanf("%d%d",&n,&k),k++;
	for(int i=1;i<n;i++){
		scanf("%d%d%d",&x,&y,&z);
		v[x].push_back({y,z}),v[y].push_back({x,z});
	}
	dfs(1,0);
	int l=-1e8,r=1e8;
	while(l<=r){
		int mid=(l+r)/2;
		c=mid,dfs(1,0);
		if(g[1].x<=k) ans=g[1].f-1ll*k*c,l=mid+1;
		else r=mid-1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

##### 5. CF802O April Fools' Problem (hard)（\*2900）

2022.7.4 同 [CF802N April Fools' Problem (medium)](https://www.luogu.com.cn/problem/CF802N)、[P4694 [PA2013] Raper](https://www.luogu.com.cn/problem/P4694)

> $n$ 天，第 $i$ 天可以花费 $a_i$ 准备一道题，花费 $b_i$ 打印一道题，每天最多准备一道，最多打印一道，准备的题可以留到以后打印。求准备并打印 $k$ 道题的最小代价（要先准备后打印）。
>
> $1\leq k\leq n\leq 5\times 10^5$，$1\leq a_i,b_i\leq 10^9$。

由于要求恰好打印 $k$ 题，考虑 wqs 二分，设 $F(i)$ 表示打印 $i$ 道题的最小代价。$F$ 是下凸的。$G(i)\gets F(i)-c\cdot i$，随着 $c$ 的增大，最优的 $i$ 也会增大。

每道题 (准备, 打印) 对应一组匹配 $(a_i,b_j)$。

考虑贪心：从左到右扫描 $b$，每个 $b_j$ 匹配前面最小且还没有被用掉的 $a_i$，用堆维护。但这样是错误的，假设此时枚举到 $b_k$，可能之前有一组 $(a_i,b_j)$ 改为 $(a_i,b_k)$ 更优，贡献 $b_k-b_j$。这启发我们采用反悔贪心。

考虑如何返回：每个 $b_i$ 要么不选；要么和 $a_j$ 匹配，贡献 $a_j+b_i-c$；要么将某个不优 $b_j$ 顶替掉，贡献 $b_i-b_j$。将 $b_i$ 提出，发现我们要找的就是 $a_j-c$ 和 $-b_j$ 的最小值。注意还需记录堆中的每个数形如 $a_j-c$ 还是 $-b_j$，从而计算打印好的题目数量。

具体地，维护一个小根堆，从小到大枚举 $i$，把 $a_i-c$ 丢入小根堆中。然后考虑当前的 $b_i$ 与堆顶匹配，如果 $b_i$ + 堆顶 $\leq 0$，就匹配上这一组，并把 $-b_i$ 丢入堆中，方便以后反悔。

时间复杂度 $\mathcal O(n\log n\log w)$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=5e5+5;
int n,k,a[N],b[N],t;
ll sum,ans;
priority_queue<pair<ll,int>,vector<pair<ll,int> >,greater<pair<ll,int> > >q;
bool ok(int c){
	sum=t=0;
	while(q.size()) q.pop();
	for(int i=1;i<=n;i++){
		q.push({a[i]-c,0});
		auto x=q.top();
		if(x.first+b[i]<=0) t+=!x.second,sum+=x.first+b[i],q.pop(),q.push({-b[i],1});
	}
	if(t>=k) ans=sum+1ll*c*k;
	return t>=k;
}
signed main(){
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++) scanf("%d",&a[i]);
	for(int i=1;i<=n;i++) scanf("%d",&b[i]);
	int l=0,r=2e9;
	while(l<=r){
		int mid=(0ll+l+r)/2;	//注意 l+r 可能爆 int，例如 l=1e9+1,r=2e9
		if(ok(mid)) r=mid-1;
		else l=mid+1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

此题的弱化版 CF802N April Fools' Problem (medium) 可以网络流：对每天建一个点，对于第 $i$ 天，连 $(s,i,1,a_i)$，$(i,i+1,\infty,0)$（$i<n$），$(i,t,1,b_i)$，然后限流 $k$，求最小费用最大流即可。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=2210,M=2e4+5;
int n,k,s,t,S,T,x,cnt=1,hd[N],to[M],nxt[M],c[M],w[M],v[N],mn[N],pre[N],id[N];
long long inf=1e18,d[N],ans;
queue<int>q;
void add(int x,int y,int z,int k){
	to[++cnt]=y,nxt[cnt]=hd[x],hd[x]=cnt,c[cnt]=z,w[cnt]=k;
	to[++cnt]=x,nxt[cnt]=hd[y],hd[y]=cnt,c[cnt]=0,w[cnt]=-k;
} 
bool spfa(){
	for(int i=0;i<=T;i++) d[i]=inf,v[i]=0;
	q.push(s),d[s]=0,v[s]=1,mn[s]=1e9;
	while(q.size()){
		int x=q.front(),y;v[x]=0,q.pop();
		for(int i=hd[x];i;i=nxt[i])
			if(c[i]&&d[y=to[i]]>d[x]+w[i]){
				d[y]=d[x]+w[i],pre[y]=x,id[y]=i,mn[y]=min(mn[x],c[i]);
				if(!v[y]) v[y]=1,q.push(y);
			}
	}
	if(d[t]==inf) return 0;
	for(int i=t;i!=s;i=pre[i]) c[id[i]]-=mn[t],c[id[i]^1]+=mn[t];
	ans+=d[t]*mn[t];
	return 1;
}
signed main(){
	scanf("%d%d",&n,&k),s=0,t=n+1,S=n+2,T=n+3;
	add(s,S,k,0),add(T,t,k,0);
	for(int i=1;i<=n;i++){
		scanf("%d",&x),add(S,i,1,x);
		if(i<n) add(i,i+1,1e9,0);
	}
	for(int i=1;i<=n;i++)
		scanf("%d",&x),add(i,T,1,x);
	while(spfa());
	printf("%lld\n",ans);
	return 0;
}
```

##### 6. CF958E2 Guard Duty (medium)（\*2200）

2022.7.4

> 有 $n$ 个点 $t_i$，每个区间都以某两个点为左右端点。要求选择 $k$ 个不相交的区间，使得这些区间的长度之和最小。
>
> $2\leq 2k\leq n\leq 5\times 10^5$，$k\leq 5000$，$1\leq t_i\leq 10^9$。

相邻两个坐标相减，即求不相邻的 $k$ 个数之和的最小值。这是一个经典的反悔贪心题：见 [P3620 [APIO/CTSC 2007] 数据备份](https://www.luogu.com.cn/problem/P3620)。也可以 wqs 二分来做。

设 $F(i)$ 表示选 $i$ 个不相邻的数的答案。$G(i)\gets F(i)-c\cdot i$，随着 $c$ 的增大，最优的 $i$ 也会增大。

设 $f_i$ 表示前 $i$ 个数，选了若干个不相邻的数的答案，以及此时选了多少个数。

时间复杂度 $\mathcal O(n\log w)$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=5e5+5;
int k,n;
ll a[N],ans;
struct dp{
	int x; ll f;
	dp operator+(dp a){return {x+a.x,f+a.f};}
	bool operator<(dp a)const{return f!=a.f?f<a.f:x>a.x;}
}f[N],t;
bool ok(int c){
	for(int i=1;i<=n;i++){
		f[i]=min(f[i-1],t={1,a[i]-c});
		if(i>=2) f[i]=min(f[i],f[i-2]+t);
		if(i>=3) f[i]=min(f[i],f[i-3]+t);
	}
	if(f[n].x>=k) ans=f[n].f+1ll*c*k;
	return f[n].x>=k;
}
signed main(){
	scanf("%d%d",&k,&n);
	for(int i=1;i<=n;i++) scanf("%lld",&a[i]);
	sort(a+1,a+1+n),n--;
	for(int i=1;i<=n;i++) a[i]=a[i+1]-a[i];
	int l=0,r=1e9;
	while(l<=r){
		int mid=(l+r)/2;
		if(ok(mid)) r=mid-1;
		else l=mid+1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

##### 7. P4983 忘情

2022.7.4

> 一个长度为 $n$ 的序列 $x$ 代价为 $\dfrac{((\sum_{i=1}^n x_i\times \overline x)+\overline x)^2}{\overline{x}^2}$。
>
> 给出一个长度为 $n$ 的序列 $a$，将其分为 $m$ 段，求每段代价之和的最小值。
>
> $m\leq n\leq 10^5$，$1\leq a_i\leq 1000$。

那一坨式子化简得 $(1+\sum x_i)^2$。

设 $F(i)$ 表示分 $i$ 段的最小代价。设这次合并的两段区间和分别为 $x,y$，那么 $F(i)-F(i-1)=(x+y+1)^2-(x+1)^2-(y+1)^2=2xy-1$，由于每次选择的 $xy$ 肯定越来越大，所以 $F(i)-F(i-1)$ 是非严格单调递增的，$F$ 是下凸的。

由于题目要求恰好 $m$ 段，考虑 wqs 二分。$G(i)\gets F(i)+c\cdot i$，那么随着 $c$ 的增大，最优的 $i$ 也会越来越小。

设 $f_i$ 表示前 $i$ 个数分为若干段的最小代价，以及此时分了多少段。求出 $a$ 的前缀和 $s$，$f_i=\min\limits_{0\leq j<i}(f_j+(s_i-s_j+1)^2-c)$。斜率优化：$\underline{f_j+s_j^2-2s_j}_{\ y}=\underline{2s_i}_{\ k}\underline{s_j}_{\ x}+\underline{f_i-s_i^2-2s_i+1}_{\ b}$，单调队列维护下凸壳即可。

时间复杂度 $\mathcal O(n\log V)$。注意 wqs 二分的斜率可以达到 $(1+\sum x_i)^2$ 即 $10^{16}$ 级别。

下面代码中，若 $f_i$ 有多个最优决策点，选最前面一个，这样算出来是“细节”中的 $l$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=1e5+5;
int n,m,q[N],g[N];
ll f[N],a[N],ans;
double get(int i){return f[i]+a[i]*a[i]-2*a[i];}
double slope(int i,int j){
	return 1.0*(get(i)-get(j))/(a[i]-a[j]);
} 
bool ok(ll c){
	int l=0,r=0;
	for(int i=1;i<=n;i++){
		while(l<r&&slope(q[l],q[l+1])<2*a[i]) l++;	//注意是 < 不是 <=，取最前面的最优决策点
		f[i]=f[q[l]]+(a[i]-a[q[l]]+1)*(a[i]-a[q[l]]+1)+c,g[i]=g[q[l]]+1;
		while(l<r&&slope(q[r-1],q[r])>=slope(q[r],i)) r--;
		q[++r]=i;
	} 
	if(g[n]<=m) ans=f[n]-c*m;
	return g[n]<=m;
}
signed main(){
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++) scanf("%lld",&a[i]),a[i]+=a[i-1];
	ll l=0,r=1e17;
	while(l<=r){
		ll mid=(l+r)/2;
		if(ok(mid)) r=mid-1;
		else l=mid+1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

##### 8. P5308 [COCI2019] Quiz

2022.7.5

> 有 $n$ 个人，每轮你会淘汰一些选手，并获得这一轮 被淘汰者数 除以 这一轮人数 元的奖金。问通过恰好 $k$ 轮使得所有选手都被淘汰，你最多能获得多少奖金。
>
> $1\leq k\leq n\leq 10^5$。

设 $F(i)$ 表示恰好 $i$ 轮的最大收益。$G(i)\gets F(i)+c\cdot i$，随着 $c$ 的增大，最优的 $i$ 也越来越大。

设 $f_i$ 表示还剩 $i$ 个人时的最大收益，同时记录此时进行了多少轮。$f_i=\max\limits_{0\leq j<i}\{f_j+\frac{i-j}{i}+c\}$，斜率优化：$\underline{f_j}_{\ y}=\underline{\frac 1 i}_{\ k}\underline{j}_{\ x}+\underline{f_i-1-c}_{\ b}$，维护一个上凸壳。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e5+5;
int n,k,g[N],q[N];
double eps=1e-12,f[N],ans; 
double slope(int i,int j){return (f[i]-f[j])/(i-j);}
bool ok(double c){
	int l=0,r=0;
	for(int i=1;i<=n;i++){
		while(l<r&&slope(q[l],q[l+1])>1.0/i) l++;
		f[i]=f[q[l]]+1.0*(i-q[l])/i+c,g[i]=g[q[l]]+1;
		while(l<r&&slope(q[r-1],q[r])<=slope(q[r],i)) r--;
		q[++r]=i;
	}
	if(g[n]<=k) ans=f[n]-c*k;
	return g[n]<=k;
}
signed main(){
	scanf("%d%d",&n,&k);
	double l=-1e6,r=0;
	while(l+eps<r){
		double mid=(l+r)/2;
		if(ok(mid)) l=mid;
		else r=mid;
	}
	printf("%.9lf\n",ans);
	return 0;
}
```

##### 9. P5896 [IOI2016]aliens

2022.7.5

> 一个 $m\times m$ 的网格图上有 $n$ 个关键点，要求选至多 $k$ 个对角线为主对角线的正方形去覆盖它们。问这些正方形最少要覆盖多少个方格。
>
> <img src="https://img2022.cnblogs.com/blog/1859218/202207/1859218-20220705141842462-1535827278.png" alt="image" style="zoom: 67%;" />
>
> $1\leq k\leq n\leq 10^5$，$1\leq m\leq 10^6$。

设 $F(i)$ 用恰好 $i$ 个正方形覆盖关键点，最少覆盖多少小方格。$G(i)\gets F(i)+c\cdot i$，当 $c=0$ 时显然最优的 $i$ 为 $n$。随着 $c$ 的增大，最优的 $i$ 会越来越小。

考虑将左上角为 $(l,l)$ 右下角为 $(r,r)$ 的正方形转化为区间 $[l,r]$。对于一个关键点 $(i,j)$，记 $u=\min(i,j),v=\max(i,j)$，则可以被 $l\leq u\leq v\leq r$ 的区间 $[l,r]$ 覆盖。

类似 P6047 丝之割，若 $(u_i,v_i),(u_j,v_j)$ 满足 $u_i\leq u_j$ 且 $v_i\geq v_j$，那么 $j$ 显然是无用的，因为覆盖 $i$ 的同时一定能覆盖 $j$。将没有的关键点去掉，按 $u$ 排序，$v$ 也单调递增。

设 $f_i$ 表示覆盖前 $i$ 个有用关键点，最少覆盖多少个方格。同时记录用了多少个区间。$f_i=\min\limits_{0\leq j<i}\{f_j+(v_i-u_{j+1}+1)^2-\max(0,v_j-u_{j+1}+1)^2+c\}$，斜率优化：$\underline{f_j+u_{j+1}^2-2u_{j+1}-\max(0,v_j-u_{j+1}+1)^2}_{\ y}=\underline{2v_i}_{\ k}\underline{u_{j+1}}_{\ x}+\underline{f_i-v_i^2-2v_i-1-c}_{\ b}$。

时间复杂度 $\mathcal O(n\log m^2)$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=1e5+5;
int n,m,k,cnt,g[N],q[N];
ll f[N],ans;
struct seg{int l,r;}a[N];
ll sq(int x){return 1ll*x*x;}
ll get(int i){
	return f[i]+sq(a[i+1].l)-2*a[i+1].l-sq(max(0,a[i].r-a[i+1].l+1));
}
double slope(int i,int j){
	return 1.0*(get(i)-get(j))/(a[i+1].l-a[j+1].l);
}
bool ok(ll c){
	int l=0,r=0;
	for(int i=1;i<=n;i++){
		while(l<r&&slope(q[l],q[l+1])<2*a[i].r) l++;
		f[i]=f[q[l]]+sq(a[i].r-a[q[l]+1].l+1)-sq(max(0,a[q[l]].r-a[q[l]+1].l+1))+c,g[i]=g[q[l]]+1;
		while(l<r&&slope(q[r-1],q[r])>=slope(q[r],i)) r--;
		q[++r]=i;
	}
	if(g[n]<=k) ans=f[n]-c*k;
	return g[n]<=k;
}
signed main(){
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++){
		scanf("%d%d",&a[i].l,&a[i].r);
		if(a[i].l>a[i].r) swap(a[i].l,a[i].r);
	}
	sort(a+1,a+1+n,[](seg x,seg y){return x.l!=y.l?x.l<y.l:x.r>y.r;});
	for(int i=1,r=-1;i<=n;i++)
		if(a[i].r>r) r=a[i].r,a[++cnt]=a[i];
	n=cnt,a[0].r=-1;
	ll l=0,r=1ll*m*m;
	while(l<=r){
		ll mid=(l+r)/2;
		if(ok(mid)) r=mid-1;
		else l=mid+1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

##### 10. P4698 [CEOI2011]Hotel

2022.7.5

> 有 $n$ 个房间，有使用成本 $c_i$ 和容量 $p_i$。还有 $m$ 个订单，有售价 $v_i$ 和人数 $d_i$。一个房间最多只能给一个订单。 求接受 $o$ 个订单时的最大收益。
>
> $1\leq n,m\leq 5\times 10^5$，$1\leq c_i,p_i,v_i,d_i\leq 10^9$，保证 $\forall 1\leq i,j\leq n$，若 $p_i<p_j$，则 $c_i\leq c_j$。

设 $F(i)$ 表示接受 $i$ 个订单的最大收益。$G(i)\gets F(i)-c\cdot i$，当 $c=0$ 时最优的 $i$ 显然是能选的全选，随着 $c$ 的增大，最优的 $i$ 也会越来越小。

将房间和订单都按人数从小到大排序，然后依次枚举每个房间，双指针求有哪些能容纳的订单，由于房间容量小的成本小，所以直接选能容纳的售价最高的那个订单即可。

```cpp
//O2
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=5e5+5;
int n,m,k,t;
ll sum,ans;
struct node{int w,num;}a[N],b[N];
priority_queue<int>q;
bool ok(int c){
	while(q.size()) q.pop();
	t=sum=0;
	for(int i=1,j=1;i<=n;i++){
		while(j<=m&&b[j].num<=a[i].num) q.push(b[j++].w);
		if(q.size()){
			int x=q.top()-a[i].w-c;
			if(x>0) t++,sum+=x,q.pop();
		}
	}
	if(t<=k) ans=sum+1ll*c*k;
	return t<=k;
}
signed main(){
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++) scanf("%d%d",&a[i].w,&a[i].num);
	for(int i=1;i<=m;i++) scanf("%d%d",&b[i].w,&b[i].num);
	sort(a+1,a+1+n,[](node x,node y){return x.num!=y.num?x.num<y.num:x.w<y.w;});
	sort(b+1,b+1+m,[](node x,node y){return x.num!=y.num?x.num<y.num:x.w>y.w;});
	int l=0,r=1e9;
	while(l<=r){
		int mid=(l+r)/2;
		if(ok(mid)) r=mid-1;
		else l=mid+1;
	}
	printf("%lld\n",ans);
	return 0;
}
```

##### [待补] 11. ZR#2183. 【2022省选十连测 Day 3】baseline

[sol](https://blog.csdn.net/weixin_43960287/article/details/123908772?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167351543516800217083337%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=167351543516800217083337&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-4-123908772-null-null.blog_rank_default&utm_term=%E5%87%B8%E5%8C%85)

##### [鸽] 12. P5617 [MtOI2019]不可视境界线

> 给出 $n$ 个半径为 $r$ 的圆，圆心为 $(x_i,0)$，求其中 $k$ 个圆的面积并最大是多少。
>
> $n,k\leq 10^5$，$r\leq 10^4$，$0\leq x_i\leq 10^9$，$x_i$ 为整数且不重复，保证 $x_i$ 单调递增。

