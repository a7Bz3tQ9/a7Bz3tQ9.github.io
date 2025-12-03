##### P3773 [CTSC2017]吉夫特

Lucas 定理 + 类似 UOJ#549 的平衡查询修改小技巧

> 给出一个长度为 $n$ 序列 $a$，求有多少长度 $\geq 2$ 的不上升子序列 $a_{b_1},a_{b_2},\cdots,a_{b_k}$ 满足：
> $$
> \prod_{i=2}^k\binom{a_{b_{i-1}}}{a_{b_i}}\bmod 2>0
> $$
> $1\leq n\leq 211985$，$1\leq a_i\leq 233333$，$a_i$ 互不相同。

根据 Lucas 定理，$\binom{a_{b_{i-1}}}{a_{b_i}}\bmod 2=1$ 当且仅当 $a_{b_{i-1}}\supseteq a_{b_i}$。

设 $f_i$ 表示以 $a_i$ 结尾的子序列个数。不管是 $f_i\to f_j$ 还是 $f_i\gets f_j$ 的方式都要枚举 $a_i$ 的子集或超集。

时间复杂度 $\mathcal O(3^{\log_2 A})$。

考虑将两种平衡一下，一半枚举子集，一半枚举超集：设 $g_{p,x,y}$ 表示前 $p$ 个数满足 $a_i$ 前 $9$ 位是 $x$、后 $9$ 位是 $y$ 的超集的 $f_i$ 之和。$p$ 是为了更清楚地体现 DP 数组的版本，实现时可以去掉。

记 $x(i)$ 表示 $i$ 的前 $9$ 位，$y(i)$ 表示 $i$ 的后 $9$ 位，那么 $f_i=\sum_{j\supseteq x(a_i)}g_{i-1,j,y(a_i)}$。求出 $f_i$ 后，对于 $y(a_i)\supseteq j$ 的 $g_{i,x(a_i),j}$ 加上 $f_i$。

#### 矩阵加速

##### B2

P5188 [COCI2009-2010#4] PALACINKE

> $n$ 个点 $m$ 条边的有向图，每条边的边权为 $\{B,J,M,P\}$ 的子集，每个字母都是一个商品。
>
> 直接经过一条边花费 $1$ 的时间，把边上的商品都买了花费 $2$ 的时间。
>
> 求在 $T$ 时间内从点 $1$ 出发回到 $1$，且 $B,J,M,P$ 这 $4$ 种上面都买过的方案数 $\bmod 5557$。
>
> $1\leq n\leq 25$，$1\leq m\leq 500$，$1\leq T\leq 10^9$。

设 $f_{t,x,S}$ 表示在时间 $t$ 走到点 $x$，买过集合 $S$ 中商品的方案数。$f_t$ 只与 $f_{t-1},f_{t-2}$ 有关，可以矩阵快速幂，但转移矩阵过大。

考虑容斥，枚举 $S$，表示 $S$ 中的商品没买，其余商品任意，如果求出 $res_S$ 表示对应方案数，那么 $ans=(-1)^{|S|}res_S$。这样，只需设 $f_{t,x}$ 表示在时间 $t$ 走到 $x$，强制 $S$ 中的商品买不了的方案数（即如果一条边能买的商品和 $S$ 有交，这条边就没有从 $f_{t-2}$ 的转移），转移矩阵就是 $n\times n$ 的了。

（弯路：没想到容斥）

（技巧：通过容斥，根据合法方案的条件，将记录状态转化成强制状态，以去掉 DP 中状态一维）

设 $ans_t$ 表示在 $\leq t$ 时间从 $1$ 出发回到 $1$ 的方案数，$ans_t=ans_{t-1}+f_{t,1}$。矩阵再开一列记 $ans$。由于同一层的 $f$ 转移到同一层的 $ans$ 不太好做，所以可以：

- 方法 1：把 $f_{t-1},f_{t-2}$ 用来算 $f_{t,1}$ 带进 $f_{t,1}$，即 $ans_t=ans_{t-1}+\sum_{(y,1)\in E,S\cap w(y,1)=\varnothing}(f_{t-1,y}+f_{t-2,y})$。
- 方法 2：$ans'_t\gets ans_{t-1}$，这样 $ans'_t=ans'_{t-1}+f_{t-1,1}$ 就可以转移了。这样要多矩乘一次。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=55,mod=5557;
int n,m,t,x,y,len,tot,id[N][3],mp[N][N],ans;
char c[5],a[4]={'B','J','M','P'};	//某 sb 刚开始写成 c[4] 数组开小了
struct mat{
	int x[N][N];
	mat(){memset(x,0,sizeof(x));}
	friend mat operator*(mat a,mat b){
		mat c;
		for(int i=1;i<=tot;i++)
			for(int j=1;j<=tot;j++)
				for(int k=1;k<=tot;k++) c.x[i][j]=(c.x[i][j]+1ll*a.x[i][k]*b.x[k][j]%mod)%mod;
		return c;
	}
};
mat qpow(mat x,int n){
	mat ans;
	ans.x[1][1]=1;
	for(;n;n>>=1,x=x*x) if(n&1) ans=ans*x;
	return ans;
}
signed main(){
	scanf("%d%d",&n,&m),memset(mp,-1,sizeof(mp));
	for(int i=1;i<=m;i++){
		scanf("%d%d%s",&x,&y,c+1),len=strlen(c+1),mp[x][y]=0;
		for(int j=1;j<=len;j++)
			for(int k=0;k<4;k++) mp[x][y]|=(c[j]==a[k])<<k;
	}
	for(int i=1;i<=n;i++) id[i][1]=++tot,id[i][2]=++tot;
	tot++,scanf("%d",&t);	//新增的第 tot 个位置表示 ans
	for(int s=0;s<(1<<4);s++){
		mat tmp;
		tmp.x[1][tot]=tmp.x[tot][tot]=1;
		for(int i=1;i<=n;i++) tmp.x[id[i][1]][id[i][2]]=1;
		for(int i=1;i<=n;i++)
			for(int j=1;j<=n;j++) if(~mp[i][j]){
				tmp.x[id[i][1]][id[j][1]]=1;	//花 1 的时间，不买
				if(!(mp[i][j]&s)) tmp.x[id[i][2]][id[j][1]]=1;	//花 2 的时间，买：要求 s 里的商品买不了
			}
		ans=(ans+1ll*(__builtin_popcount(s)&1?mod-1:1)*qpow(tmp,t+1).x[1][tot]%mod)%mod;	//多矩乘一次
	}
	printf("%d\n",ans);
	return 0;
}
```

#### 单调队列优化

形如 $f_i=\max\limits_{j=l_i}^{i-1}\{g_j\}+w_i$，$l_i\leq l_{i+1}$。对于 $i_1<i_2$，$g_{i_1}$ 没有 $g_{i_2}$ 优秀，那么之后都用不到 $i_1$ 转移了。

维护一个队列，下标递增，元素从优到劣。每次若队尾没有新加入的元素优秀就出队，这样队首就一直保留最优转移，但若队首超出了区间也要弹出队列。每次转移就直接取队首。

##### C1

CF372C Watching Fireworks is Fun（\*2100）

> 有 $n$ 个位置，$m$ 个烟花要放，第 $i$ 个烟花放出时间为 $t_i$，放出位置为 $a_i$。若放出时你在 $x$，将会收获 $b_i-|a_i-x|$ 点快乐值。
>
> 初始时你可以在任意位置，每个单位时间你可以移动不超过 $d$ 个单位距离，最大化快乐值。
>
> $1\leq n\leq 1.5\times 10^5$，$1\leq m\le 300$，$1\leq b_i,t_i\leq 10^9$，$t_i\leq t_{i+1}$。

设 $f_{i,j}$ 表示放第 $i$ 个烟花时，位置在 $j$ 的最大快乐值。记 $r_i=(t_i-t_{i-1})d$，$f_{i,j}=\max\limits_{k=j-r_i}^{j+r_i}\{f_{i-1,k}\}+b_i-|a_i-j|$。固定 $i$，$j-r_i,j+r_i$ 都是递增的，单调队列优化即可。

时间复杂度 $\mathcal O(nm)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1.5e5+5;
int n,m,d,a[N],b[N],t[N],l,r,q[N],cur=1,pre;
long long x,f[2][N],ans=-1e18;
signed main(){
	scanf("%d%d%d",&n,&m,&d);
	for(int i=1;i<=m;i++){
		scanf("%d%d%d",&a[i],&b[i],&t[i]);
		swap(pre,cur),l=1,r=0,x=1ll*(t[i]-t[i-1])*d;	//注意初始不是 l=r=0！
		for(int j=1,k=1;j<=n;j++){
			while(k<=min(j+x,1ll*n)){
				while(l<=r&&f[pre][k]>f[pre][q[r]]) r--;
				q[++r]=k++;
			}
			while(l<=r&&q[l]<j-x) l++;
			f[cur][j]=f[pre][q[l]]+b[i]-abs(a[i]-j); 
		}
	}
	for(int i=1;i<=n;i++) ans=max(ans,f[cur][i]);
	printf("%lld\n",ans);
	return 0;
}
```

#### 背包优化

##### D1

> $n$ 个物品，每个物品重量为 $w_i$，价值为 $v_i$，数量为 $k_i$。背包承重不超过 $m$，求最大价值。

**二进制拆分优化：**

对于第 $i$ 个物品，将 $2^0,2^1,2^2,\cdots,2^c,k_i-2^{c+1}+1$ 个物品分别看成一个物品后变成 $01$ 背包（$2^0+\cdots+2^c=2^{c+1}-1$，数量拆掉 $2^0,\cdots,2^c$ 后剩下 $k-(2^{c+1}-1)$ 个）。

正确性：不管最终选了几个，都能根据二进制用这些物品凑出来。因为 $2^0,\cdots,2^c$ 可以凑出 $[0,2^{c+1})$，$2^0,\cdots,2^c$ 再拼上 $k_i-2^{c+1}+1$ 就可以凑出 $[k_i-2^{c+1}+1,k_i]$，而 $k_i-2^{c+1}+1\leq 2^{c+1}$。

时间复杂度 $\mathcal O(nm\log k)$。

**单调队列优化：**

设 $f_{i,j}$ 表示前 $i$ 个物品选出的总重量为 $j$ 时的最大价值，$f_{i,j}=\max\limits_{p=0}^{k_i}\{f_{i-1,j-w_ip}+v_ip\}$。

按模 $w_i$ 的余数分开，每一个都可以用单调队列优化。记 $g_{i,x,y}=f_{i,w_ix+y}$，那么 $g_{i,x,y}=\max\limits_{p=0}^{k_i}\{g_{i-1,x-p,y}+v_ip\}$，$g_{i,x,y}=\max\limits_{p=0}^{k_i}\{g_{i-1,x-p,y}-(x-p)v_i\}+xv_i$。固定 $i,y$，$x-k_i$ 递增。

（技巧：尽量写成 $f_j+w(j,i)$ 的形式）

时间复杂度 $\mathcal O(nm)$。

##### D2

CF1428G2 Lucky Numbers (Hard Version)（\*3000）

> 给定 $k,F_0,F_1,F_2,F_3,F_4,F_5$。
>
> 你要把数 $n$ 变成 $k$ 个数的和，对 $k$ 个数中的每一个数，若第 $i$ 位是 $3$ 的 $u$ 倍就会获得 $uF_i$的价值（第 $i$ 位是 $0,3,6,9$ 分别获得 $0,F_i,2F_i,3F_i$）。
>
> 你要让价值最大，求最大价值。多组询问 $q$ 次（$k,F_i$ 不变，$n$ 变）。
>
> $1\leq q\leq 10^5$，$1\leq n,k<10^6$，$1\leq F_i\leq 10^9$。

$n<10^6$ 说明最多 $6$ 位。

考虑每一位，$k$ 个数中最多只有一个数在这一位不是 $3$ 的倍数。调整法证明：对于 $x,y\not\in\{0,3,6,9\}$，若 $x+y\leq 9$，改为 $0$ 和 $x+y$；若 $x+y>9$，改为 $9$ 和 $x+y-9$。

设 $k$ 个数在这一位上分别是 $x_1,\cdots,x_k$。我们把每位不是 $3$ 的倍数的那个都放到 $x_k$。前 $k$ 个数直接做多重背包。

具体地，那么对于第 $i$ 位，$x_1,\cdots,x_{k-1}$ 都是 $3$ 的倍数。把 $6$ 看成 $2$ 个 $3$，$9$ 看成 $3$ 个 $3$，我们不关心放了 $3,6,9$ 的哪个，只关心放了几个 $3$。就算 $x_{1\sim k-1}$ 都是 $9$，最多也只能放 $3(k-1)$ 个 $3$。可以看成有 $6$ 个物品，第 $i$ 个物品 **重量 $3\cdot 10^i$，价值 $F_i$，个数 $3(k-1)$**。二进制优化多重背包即可。

最后考虑 $x_k$。用分组背包把最后一个数的每一位都加进去，$0\sim 9$ 都能选，但只有 $3,6,9$ 才有贡献。不过可以直接枚举第 $k$ 个数，初始化 DP 数组，再多重背包前 $k-1$ 个数。

时间复杂度 $\mathcal O(n\log k)$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=1e6+5;
int k,q,n=1e6,a[7];
ll f[N];
void add(int w,ll v){
	for(int i=n;i>=w;i--) f[i]=max(f[i],f[i-w]+v);
}
void ins(int w,int v,int c){
	int k=0;
	c=min(c,n/w);	//!!!
	while(c>=(1<<k)) add(w*(1<<k),1ll*v*(1<<k)),c-=(1<<k),k++; k--;
	if(c) add(w*c,1ll*v*c);
}
signed main(){
	scanf("%d",&k);
	for(int i=0;i<6;i++) scanf("%d",&a[i]);
	for(int i=1;i<=n;i++){	//枚举第 k 个数 i，算出它的价值 f[i]
		int x=i;
		for(int j=0;j<6;j++){
			int v=x%10; x/=10;
			if(v%3==0) f[i]+=1ll*(v/3)*a[j];
		}
	}
	for(int i=0,pw=1;i<6;i++,pw*=10) ins(3*pw,a[i],3*(k-1));	//多重背包
	scanf("%d",&q);
	while(q--)
		scanf("%d",&n),printf("%lld\n",f[n]);
	return 0;
}
```

##### D3

P3403 跳楼机，同余最短路

> 有一栋 $h$ 层楼的房子，电梯可以：1. 向上移动 $x$ 层；2. 向上移动 $y$ 层；3. 向上移动 $z$ 层；4. 回到第一层。
>
> 你现在在第一层，求可以到达的楼层数。
>
> $1\leq h<2^{63}$，$1\leq x,y,z\leq 10^5$。

重要性质：若楼层 $i$ 可达，那么 $i+x,i+2x,i+3x,\cdots$ 也都可达。

设 $f_i$ 表示仅用 2,3 操作能到达的 $\bmod x=i$ 的最小的 $i$。$f_i+y\to f_{(i+y)\bmod x}$，$f_i+z\to f_{(i+z)\bmod x}$。建图跑最短路后得到 $f_i$，$ans=\sum_{i=0}^x(\lfloor\frac{h-f_i}{x}\rfloor+1)$。

时间复杂度 $\mathcal O(x\log x)$。

#### 分治优化

##### E1

CF1442D Sum（\*2800）

> $n$ 个不降的数组，如下操作 $k$ 次：
>
> - 选择一个非空数组，把第一个元素计入价值，然后删除第一个元素。
>
> 求最大价值。元素总个数 $\leq 10^6$，$n,k\leq 3000$。

发现取了没取满的数组最多有一个。若有两个，肯定是让当前取的结尾小的那个少取点，让另一个取满。

需要计算去掉第 $i$ 个数组，剩下 $n-1$ 个数组的背包，最后再插入第 $i$ 个数组。分治算缺一背包即可。

时间复杂度 $\mathcal O(nk\log n)$。

##### E2

P4093 [HEOI2016/TJOI2016]序列

> 长度为 $n$ 的序列 $a$，可能发生 $m$ 种变化，每一种变化 $(x_i,y_i)$ 对应着 $a_{x_i}\gets {y_i}$。
>
> 求在任意一种变换中都不降的最长的子序列，输出长度。
>
> $n,m,a_i\leq 10^5$。

设 $a_i$ 最大、最小能变成的值为 $mx_i,mn_i$，$f_i$ 表示以 $i$ 结尾的最长子序列长度。

$f_i=\max\limits_{j<i,a_j\leq mn_i,mx_j\leq a_i}\{f_j+1\}$。三维偏序，CDQ 分治优化即可。

（从单点分析，因为一个方案在单点不合法就一定不合法。任意变换中 $a_j\leq a_i$ $\Leftrightarrow$  $a_j\leq mn_i\land mx_j\leq a_i$）

时间复杂度 $\mathcal O(n\log^2 n)$。

#### 长链剖分优化 DP

##### F1

> 求树上长度 $\leq k$ 的路径条数。
>
> $n\leq 10^6$。

对于一条路径，在 $\text{lca}$ 处统计它的贡献。

继承重儿子，加入轻儿子。

对每个轻儿子 $y$，枚举 $i\leq mx_y$，将 $\sum_{j+i+1\leq k}f_{x,j}\times f_{y,i}$ 加入答案，统计完答案后再 $f_{x,i+1}\gets f_{x,i+1}+f_{y,i}$。要求 $f_{x,0\sim k-i-1}$ 的和，前缀和看似要线段树，实际上可以改为计算 总数 - 后缀和，维护后缀和。每次改一个位置，就是改了后缀和的一段前缀，枚举量是 $mx_y$ 的，和加入 $f_y$ 复杂度一样了。

时间复杂度 $\mathcal O(n)$。

##### F2

P3565 [POI2014]HOT-Hotels

> 给出一棵树，在树上选 $3$ 个点，要求两两距离相等，求方案数。
>
> $n\leq 10^6$。

这三个点肯定存在一个中心点，中心点到三个点的距离相等。

我们还是希望在三个点的 $\text{lca}$ 处统计答案。

设 $f_{x,i}$ 表示 $x$ 子树里有多少距离 $x$ 为 $j$ 的点，$g_{x,i}$ 表示 $x$ 子树中，两个点到它们的 $\text{lca}$ 距离为 $d$，且 $\text{lca}$ 到 $x$ 的距离为 $d-i$ 的点对数。

一种情况是 $g_{x,0}$，另一种：

<img src="https://img2022.cnblogs.com/blog/1859218/202208/1859218-20220802170055865-1074829287.png" alt="image" style="zoom:50%;" />

#### 杂

##### F1

P7606 [THUPC2021] 混乱邪恶

> 有 $n$ 个道具和一个无限大的三角形网格图，初始在原点，$L=G=0$。
>
> 每个道具能让你往 $6$ 个方向中的一个走一步，然后 $L,G$ 分别在模 $p$ 意义下加上一个数，道具不同方向不同那么加的数也不同。
>
> 你需要选择每个道具走的方向，使得 $n$ 个道具使用完后可以回到原点且 $L,G$ 为给定的值。问是否存在一种合法方案。
>
> $n,p\leq 100$。

建二维直角坐标系，$6$ 个方向为：$x$ 轴、直线 $y=x$、$y$ 轴（把三角形拉直）。

<img src="https://img2022.cnblogs.com/blog/1859218/202207/1859218-20220725214451996-1585818239.png" alt="image" style="zoom:60%;" />

设 $f_{i,j,k,l,g}$ 表示用了前 $i$ 个道具，现在在点 $(j,k)$，当前的 $L$ 为 $l$，当前的 $G$ 为 $g$，整个状态是否可达。

直接暴力背包做，时间复杂度 $\mathcal O(n^3p^2)$。可以使用 `bitset` 压位优化，把 $y$ 压一下（压别的也可以），相当于右移再或起来，时间复杂度 $\mathcal O(\frac{n^3p^2}{w})$，还是过不了。

<img src="https://img2022.cnblogs.com/blog/1859218/202207/1859218-20220725214631576-1522234960.png" alt="image" style="zoom:50%;" />

最后合法的方案，随机打乱后不会离原点太远，可以证明是 $\mathcal O(\sqrt n)$ 的。

比如在数轴上，一个合法方案 $+1,+1,\cdots,+1,-1,-1\cdots,-1$，shuffle 之后 $+1,-1$ 的概率都是 $\frac 1 2$，离原点不会太远。具体证明：设 $x$ 的势能是 $x^2$，各有 $\frac 1 2$ 的概率走到 $x+1,x-1$，得到 $\frac 1 2[(x+1)^2+(x-1)^2]=x^2+1$，即走一步期望势能 $+1$。那么走了 $n$ 步势能期望是 $n$，而势能又是距离的平方，所以距离期望 $\sqrt n$。

所以随机道具的顺序之后，把 DP 状态里的 $j,k$ 都变成 $\sqrt n$ 大小，实际题目中开 $20$ 就行了。

时间复杂度 $\mathcal O(\frac{n^2p^2}{w})$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=110,M=25;
int n,p,L,G,o=12,a[N][7],b[N][7],id[N],dx[6]={1,1,0,-1,-1,0},dy[6]={1,0,-1,-1,0,1},cur=1,pre;
bitset<N>f[2][M][M][N],tmp;
signed main(){
	scanf("%d%d",&n,&p);
	for(int i=1;i<=n;i++){
		id[i]=i;
		for(int j=0;j<6;j++) scanf("%d%d",&a[i][j],&b[i][j]);
	}
	scanf("%d%d",&L,&G);
	random_shuffle(id+1,id+1+n),f[cur][o][o][0].set(0);
	for(int i=1;i<=n;i++){
		swap(pre,cur);
		for(int x=-o;x<=o;x++)
			for(int y=-o;y<=o;y++)
				for(int l=0;l<p;l++) f[cur][x+o][y+o][l].reset();
		for(int x=-o;x<=o;x++)
			for(int y=-o;y<=o;y++)
				for(int l=0;l<p;l++) if((tmp=f[pre][x+o][y+o][l]).any())
					for(int k=0;k<6;k++){
						int nx=x+dx[k],ny=y+dy[k],tb=b[id[i]][k];
						if(nx<-o||nx>o||ny<-o||ny>o) continue;
						f[cur][nx+o][ny+o][(l+a[id[i]][k])%p]|=(tmp<<tb)|(tmp>>(p-tb));
					}
	}
	puts(f[cur][o][o][L].test(G)?"Chaotic Evil":"Not a true problem setter");
	return 0;
} 
```

##### F2

CF1450G Communism（\*3500）

> 有 $n$ 个工人站成一排，第 $i$ 个人的工作为 $s_i$（用长度为 $n$ 的字符串表示）。有理数 $k=\frac a b$ 作为参数。
>
> 每次操作选择一个至少有一个工人的工作 $x$，设所有工作为 $x$ 的工人的位置为 $i_1,\cdots,i_m$，若 $k\cdot (i_m-i_1+1)\leq m$，则可以再选另一个至少有一个工人的工作 $y$，将所有工作为 $x$ 的人的工作换成 $y$。
>
> 若可以通过若干次（含 $0$ 次）操作将所有人的工作替换成 $x$，则称工作 $x$ 是可达的。输出可达的工作个数，并将它们按字典序输出。
>
> $1\leq n\leq 5000$，字符集大小为 $20$。

考虑如果工作为 $x$ 的都能替换成 $y$，那么就连一条边 $x\to y$，这样能形成一棵树，儿子能变成父亲，然后不断向上变。这方便我们设计 DP 状态。

只有 $20$ 种字母，考虑状压 DP，每种字符提出第一次出现 $i_1$，最后一次出现 $i_m$ 和出现个数 $m$ 就可以不管原来的字符串了。设 $f_S$ 表示集合 $S$ 里的字符能否转移成一个新的字符。

- 若干子集会分别变成这个字符。也就是把森林合并成树：简单地合并，$f_S\gets f_T\and f_{S\backslash T}$，因为 $T,S\backslash T$ 都能转移成一种字符，只要让它们转移的字符是同一个即可。记 $C$ 是字符集大小，时间复杂度 $\mathcal O(3^C)$。
- 先转移成一个再一起转移。只有一棵树，给这棵树封顶（每次在树顶上加一个点再连出去一条边，连出去一条边是为了转移到下次封顶的情况）：首先加的顶一定在树里，$i\in S$，并且去掉这个顶合法，$f_{S-\{i\}}=1$。时间复杂度 $\mathcal O(2^CC)$。

最后字符 $i$ 合法等价于 $f_{U-\{i\}}=1$。优化第一种转移。

记 $s$ 所涵盖的区间为 $range(s)$（$s$ 中的字符，最左边出现的位置，最右边出现的位置，组成区间）。那么只有 $range(T)\cap range(S\backslash T)=\varnothing$ 时才用第一种转移。否则，若存在相交的 $s_1,s_2$，可以先把 $s_1$ 转移的字符变成 $s_2$ 转移的字符，剩下的一定合法：$k\cdot len(range(s_1\cup s_2))\leq k\cdot len(range(s_1))+k\cdot len(range(s_2))\leq |s_1|+|s_2|\leq |s_1\cup s_2|$。

时间复杂度 $\mathcal O(2^CC+n)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=5e3+5,M=27;
int n,a,b,id[M],tot,l[M],r[M],c[M],f[1<<20],cnt;
char s[N],ans[N];
signed main(){
	scanf("%d%d%d%s",&n,&a,&b,s+1);
	fill(id,id+26,-1),f[0]=1;
	for(int i=1;i<=n;i++){
		int k=s[i]-'a';
		if(!~id[k]) l[id[k]=tot++]=i;
		r[id[k]]=i,c[id[k]]++;
	}
	for(int s=1;s<(1<<tot);s++){
		int L=n,R=0,m=0,t=0,ok=0;
		for(int i=0;i<tot;i++) if(s>>i&1){
			t|=1<<i,f[s]|=f[t]&&f[s^t];	//第一种转移（虽然好像和题解写的对不上捏）
			L=min(L,l[i]),R=max(R,r[i]),m+=c[i];
			ok|=f[s^(1<<i)];
		}
		f[s]|=(ok&&a*(R-L+1)<=b*m);	//第二种转移
	}
	for(int i=0;i<tot;i++)
		if(f[((1<<tot)-1)^(1<<i)]) ans[++cnt]=s[l[i]];
	sort(ans+1,ans+1+cnt),printf("%d ",cnt);
	for(int i=1;i<=cnt;i++) putchar(ans[i]),putchar(' ');
	return 0;
}
```

#### 交换值域和定义域

#####  ZR#1819. Literary Trick

2021.8.14 经典题回炉重造

> 给出两个字符串 $s,t$，若 $s,t$ 的编辑距离 $\leq k$，输出它，否则输出 `-1`。
>
> $n,m\leq 5\times 10^5$，$k\leq 5000$。

（24 省选第二轮集训听 xtq 讲 QOJ#5312. Levenshtein Distance 知道了该怎么想到 DP 状态）

设 $f_{i,j}$ 表示 $s[1:i]$ 和 $t[1:j]$ 的编辑距离。$f_{i,j}=\min(f_{i-1,j}+1,f_{i,j-1}+1,f_{i-1,j-1}+[s_i\neq t_j])$。复杂度 $\mathcal O(n^2)$。

发现若 $|i-j|>k$，那么一定有 $f_{i,j}>k$（因为每次操作至多让串长变化 $1$）。于是只要保留 $|i-j|\leq k$ 的状态。复杂度 $\mathcal O(nk)$。

考虑把 $f_{i,j}$ 摆在平面上。不难发现一条对角线上的 $f$ 值是单调的（向右或向下走都要 $1$ 的代价，向右下走要 $\leq 1$ 的代价。故 $f_{i,j}\geq f_{i-1,j-1}$），所以每条对角线都形如：依次为 若干 $0$、若干 $1$、若干 $2$、...、若干 $k$。

于是设 $g_{v,x}$ 表示 $f_{i,j}=v$ 且 $j-i=x$ 的 $i$ 的最大值。即 $j-i=x$ 这条对角线上，$f$ 值为 $v$ 的段的末尾。

假设 $(i,j)$ 是它所在对角线 $f_{i,j}$ 段的末尾（$v=f_{i,j}$，$x=j-i$）：

- 向右走一步，再在 $f$ 值不变的条件下能走多远走多远（即转移向 $(i,j+1)$ 所在对角线 $f_{i,j}+1$ 段的末尾）：$g_{v+1,x+1}\gets g_{v,x}+\text{lcp}(s[i+1:n],t[j+2:m])$。

- 向下走一步，再在 $f$ 值不变的条件下能走多远走多远（即转移向 $(i+1,j)$ 所在对角线 $f_{i,j}+1$ 段的末尾）：$g_{v+1,x-1}\gets g_{v,x}+1+\text{lcp}(s[i+2:n],t[j+1:m])$。
- 向右下走一步，再能走多远走多远：$g_{v+1,x}\gets g_{v,x}+1+\text{lcp}(s[i+2:n],t[j+2:m])$。

实现时，枚举 $v,x$，那么 $i=g_{x,v},j=i+x$。

时间复杂度 $\mathcal O(k^2)$。

#### 数据结构优化 DP

##### E1

DP 优化小练习 1

> 一棵 $n$ 个节点的树，每个点有一个重量为 $v_i$ 的物品。
>
> 对 $k\in[1,m]$ 求出，树上有多少个连通块物品重量和为 $k$，$\bmod 10^9+7$。
>
> $n\leq 200$，$m\leq 50000$。

若这个连通块要包含 $1$ 怎么做？是个树形依赖背包，在 DFS 序上跑一遍：设 $f_{i,j}$ 考虑到 DFS 序为 $i$，重量和为 $j$ 的方案数。若 $i$ 选，$i$ 子树里的点还能选，$f_{i,j}\to f_{i+1,j+v_i}$；否则，直接跳过整个子树，$f_{i,j}\to f_{out_i+1,j}$。

若 $w_i\leq 1$ 肯定能做，是经典的 $\mathcal O(n^2)$ 背包合并。

对每个 $i$ 求 $f_{i,j}$ 表示以 $i$ 为根的连通块重量和为 $j$ 的方案数。暴力背包合并复杂度 $\mathcal O(nm^2)$，无法通过。

背包合并做不了，改成一个一个加入。

类似 Dsu on tree，对一个点 $i$，可以先继承重儿子的 DP 数组，然后爆搜所有轻子树，类似树形依赖背包依次将每个物品加入背包即可，在 dfs 序上跑一遍。注意不是合并而是加入。

时间复杂度 $\mathcal O(nm\log n)$。

##### E2

DP 优化小练习 2：HDU 6566 The Hanged Man、nflsoj#1031. 【2021 六校联合训练 NOI #30】夏影，类似 ZR#2243. 22noi-day3-白杨树

> $n$ 个点的树，每个点有两种权值 $a_i,b_i$。你需要选一个独立集，使得 $a_i$ 的总和恰好为 $m$，且 $b_i$ 的总和尽量大。
>
> $n\leq 100$，$m\leq 5000$。

设 $f_{i,j,0/1}$ 表示 $i$ 的子树中 $\sum a=j$，$i$ 是否选择时，$\sum b$ 的最大值。复杂度 $\mathcal O(nm^2)$。

考虑将背包的合并，转化成插入，每次拓展一个点。但由于必须选成独立集，需要状压。注意到我们并不需要记录所有点的选择情况，只需要记录一部分。有两种做法：

- 方法 1：按 dfs 序将点依次加入背包，要记 $f_{i,j,S}$，$S$ 表示 $i$ 祖先的选择状态，因为走完子树 $i$ 后要往回走去 dfs 其他子树，这时需要 $i$ 祖先的状态。但若 $i$ 是 $fa_i$ 的最后一个子树，就不需要记 $fa_i$ 了；同理若 $fa_i$ 是 $fa_{fa_i}$ 的最后一个子树，就不需要记 $fa_{fa_i}$ 了，以此类推。

  考虑树链剖分，先 dfs 轻儿子，再 dfs 重儿子，这样就不需要记录重儿子父亲的选择状态了。只有一个点是一条轻边的父亲时才要记录状态，$S$ 变成 $2^{\log n}=n$。

  <img src="https://img2022.cnblogs.com/blog/1859218/202208/1859218-20220802155739217-95714121.png" alt="image" style="zoom: 57%;" />

- 方法 2：考虑点分治，在点分树上 DP。注意到能和一个点相邻的点要么是点分树上的祖先，要么是它子树里的点。因此 DP 时暴力记点分树上的祖先被选的情况即可。

时间复杂度 $\mathcal O(n^2m)$。

#### 斜率优化

让截距最小：三个决策点 $j_1,j_2,j_3$，若 $slope(j_1,j_2)\geq slope(j_2,j_3)$，那么 $j_2$ 就永远不会成为决策点。所以决策点一定在点集的下凸壳上。

##### A1

> 有 $n$ 个任务，第 $i$ 个任务单独完成所需时间为 $t_i$。将 $n$ 个任务分为若干组，每组包含相邻的若干任务，在每组任务开始前，机器需要启动时间 $s$，完成这组任务的时间是各个任务所需时间之和。
>
> 同一组任务将在同一时刻完成。每个任务的费用是它的完成时刻 $\times c_i$。求最小总费用。
>
> sub1：$1\leq n\leq 5000$，$s,t_i,c_i\geq 1$。
>
> sub2：$1\leq n\leq 3\times 10^5$，$s,t_i,c_i\geq 1$。
>
> sub3：$t_i$ 不一定是正数。
>
> sub4：$c_i,t_i$ 都不一定是正数。

**sub1：**P2365 任务安排

求出 $t_i,c_i$ 的前缀和 $st_i,sc_i$。

设 $f_{i,j}$ 表示前 $i$ 个任务分出 $j$ 批的最小总费用，$f_{i,j}=\min\limits_{k<i}\{f_{k,j-1}+(sc_i-sc_k)(st_i+s\cdot j)\}$。

一组任务的完成时间依赖于 $j$。考虑费用提前计算，每次重新启动时就把后面的贡献算了，这样可以去掉 $j$ 这维。

$f_i=\min\limits_{j<i}\{f_j+(sc_i-sc_j)st_i+s(sc_n-sc_j)\}$。时间复杂度 $\mathcal O(n^2)$。

**sub2：**

$\underline{f_j-s\cdot sc_j}_{\ y}=\underline{st_i}_{\ k}\underline{sc_j}_{\ x}+\underline{f_i-sc_ist_i-s\cdot sc_n}_{\ b}$。插入时，$\underline{sc_j}_{\ x}$ 是递增的，相当于每次在最右边插入，可以队列维护，若队尾不优就弹掉（维护下凸壳），然后加入新元素。查询时，$\underline{st_i}_{\ k}$ 是递增的，若队首两个点的斜率小于 $k$ 就弹出队首（因为 $k$ 递增，现在 $<k$，之后还是 $<k$）。最后直接用队首的点更新答案。

时间复杂度 $\mathcal O(n)$。

**sub3：**P5785 [SDOI2012]任务安排

$k$ 不单调了。是改为在凸包上二分。

**sub4：**

$k,x$ 都不单调了。

1. 平衡树：

   用平衡树动态维护上凸包找到插入的位置，把前后不优的点弹掉后，判断插入的点是否能插入。查询还是在凸包上二分。

2. 离线 CDQ 分治：

   分治保证按原来的顺序即 $id$ 转移，用左半区间更新右半区间（用 $id$ 小的更新 $id$ 大的）。对左区间的点按 $x$ 排序依次插入维护凸壳，用来更新右区间。对右区间已经按 $k$ 排好序的点更新答案。

3. 李超树：

   不用斜率优化，将转移表示成直线的形式，$y$ 为当前要求的 $f_i$，$x$ 为当前点 $i$ 的相关信息。这样只需要支持插入一条直线，查询给定 $x$ 所有直线 $y$ 的最值。

平衡树维护动态凸包：CF70D Professor's task（\*2700）。

> $q$ 次操作，支持：
>
> - `1 x y`：将 $(x,y)$ 插入 $S$。
> - `2 x y`：查询 $(x,y)$ 是否在 $S$ 的凸包中。
>
> $4\leq q\leq 10^5$，$-10^6\leq x,y\leq 10^6$。

考虑 Andrew 算法，分别维护上下凸壳。用 `set` 维护坐标，排序函数就是 Andrew 算法的排序函数。

查询时判断是否在上下凸壳内部：找到前驱后继，判断询问点是否在这两个点的连边右侧，若是则不在。插入时先判断是否在上下凸壳内部，若不是则插入，然后弹掉两侧多余的点。

时间复杂度 $\mathcal O(q\log q)$。

```cpp
#include<bits/stdc++.h>
using namespace std;
int q,op;
struct P{
	int x,y;
	bool operator<(P a)const{return x^a.x?x<a.x:y<a.y;}
	bool operator>(P a)const{return x^a.x?x>a.x:y>a.y;}
	P operator-(P a){return {x-a.x,y-a.y};}
	long long operator*(P a){return 1ll*x*a.y-1ll*y*a.x;}
}p;
set<P>up;
set<P,greater<P> >dn;
bool R(P x,P y,P z){return (z-x)*(z-y)>=0;}
bool qry(auto &s){
	if(s.count(p)) return 1;
	auto it=s.lower_bound(p);
	return it!=s.end()&&it!=s.begin()&&R(*prev(it),*it,p);
}
void add(auto &s){
	if(qry(s)) return ;
	auto it=s.insert(p).first;
	if(it!=s.begin())
		for(auto x=prev(it);x!=s.begin();){
			auto y=prev(x); 
			if(R(*y,*it,*x)) s.erase(x),x=y; else break;
		}
	if(it!=--s.end())
		for(auto x=next(it);x!=--s.end();){
			auto y=next(x);
			if(R(*it,*y,*x)) s.erase(x),x=y; else break;
		}
} 
signed main(){
	scanf("%d",&q);
	while(q--){
		scanf("%d%d%d",&op,&p.x,&p.y);
		if(op==1) add(dn),add(up);
		else puts(qry(dn)&&qry(up)?"YES":"NO");
	}
	return 0;
}
```

##### A6

CF1067D Computer Game（\*3100），2022.10.22 补

> 有 $n$ 个任务，第 $i$ 任务有 $p_i$ 的概率完成，收益为 $a_i$，一个任务可以做多次。
>
> 若完成了其中某个任务，可以选择对任意一个尚未升级的任务进行升级（或不升级任何任务）。任务升级后，$p_i$ 不变，单次的收益提高到 $b_i$。
>
> 你还能做 $t$ 个任务，求总收益的期望的最大值。
>
> $1\leq n\leq 10^5$，$1\leq t\leq 10^{10}$，$1\leq a_i<b_i\leq 10^8$，$0<p_i<1$。

一旦获得升级任务的机会，肯定会选 $p_ib_i$ 最大的任务进行升级，由于 $a_i<b_i$，升级后一直做这个 $p_ib_i$ 最大的任务，后面期望总收益最大。记 $m=\max p_ib_i$。

设 $f_i$ 表示当前尚未完成任何一个任务，还有 $i$ 次机会时，所得收益期望的最大值。$f_i=\max_{j=1}^n\{p_j(a_j+(i-1)m)+(1-p_j)f_{i-1}\}$，$\underline{p_ja_j}_{\ y}=\underline{(f_{i-1}-(i-1)m)}_{\ k}\underline{p_j}_{\ x}+\underline{f_i-f_{i-1}}_{\ b}$。

- 将任务按 $p_j$ 从小到大排序，这样 $x$ 是递增的。维护 $(p_j,p_ja_j)$ 的上凸壳。

- $k_{i+1}-k_i=f_i-f_{i-1}-m$，而显然 $f_i-f_{i-1}\leq m$，故 $k_{i+1}-k_i\leq 0$，切凸壳的直线斜率单减，所以随着 $i$ 的增大上凸壳上转移点的 $x$ 坐标单增。

  > 也可以贪心来看：$i$ 较大时取最大的 $p_j$ 来获取升级机会是好的，$i$ 较小时升级后的收益会被冲淡，要考虑进 $a_j$，但随着 $i$ 的增大转移点的 $p$ 只会越来越大。

故可以单调队列维护。但 $t$ 过大。

一个 new trick 是考虑每个转移点考虑了对少次，由于 $t$ 很大所以矩阵加速，倍增地跳，若跳过去还是当前转移点更优就跳，最后再补跳一步。
$$
\begin{bmatrix}f_i&i&1\end{bmatrix}
\times
\begin{bmatrix}
1-p_j&0&0\\
m\cdot p_j&1&0\\
a_j\cdot p_j&1&1
\end{bmatrix}
=
\begin{bmatrix}f_{i+1}&i+1&1\end{bmatrix}
$$

时间复杂度 $\mathcal O(n\log t)$。


```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=1e5+5;
int n,q[N],l=1,r;
ll t;
double m,eps=1e-12;
struct P{
	int a,b; double p;
}a[N];
struct mat{
	double x[3][3];
	friend mat operator*(mat a,mat b){
		mat c;
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++){
				c.x[i][j]=0;	//!!!
				for(int k=0;k<3;k++) c.x[i][j]+=a.x[i][k]*b.x[k][j];
			}
		return c;
	}
}f,g,pw[40];
double X(int i){return a[i].p;}
double Y(int i){return a[i].p*a[i].a;}
double slope(int i,int j){
	return fabs(X(i)-X(j))<eps?(Y(i)<Y(j)?1e50:-1e50):(Y(j)-Y(i))/(X(j)-X(i));
}
signed main(){
	scanf("%d%lld",&n,&t);
	for(int i=1;i<=n;i++)
		scanf("%d%d%lf",&a[i].a,&a[i].b,&a[i].p),m=max(m,a[i].b*a[i].p);
	sort(a+1,a+1+n,[](P x,P y){return x.p<y.p;});
	for(int i=1;i<=n;i++){
		while(l<r&&slope(q[r-1],q[r])<=slope(q[r],i)) r--;
		q[++r]=i;
	}
	f.x[0][2]=1;
	for(ll i=0,x;i<t;){	//不用 i++!!!
		while(l<r&&slope(q[l],q[l+1])>=f.x[0][0]-i*m) l++;
		int j=q[l];
		pw[0]={1-a[j].p,0,0,m*a[j].p,1,0,a[j].a*a[j].p,1,1};
		for(int k=1;k<=34;k++) pw[k]=pw[k-1]*pw[k-1];
		for(int k=34;k>=0;k--) if((x=i+(1ll<<k))<t){	//1ll!!!
			g=f*pw[k];
			if(l==r||slope(q[l],q[l+1])<=g.x[0][0]-x*m) i=x,f=g;
		}
		i++,f=f*pw[0];
	}
	printf("%.9lf\n",f.x[0][0]);
	return 0;
}
```

#### 凸优化-WQS 二分

解决问题：题目给了选物品的限制条件，要求恰好选 $m$ 个，最大化权值。

要求：$f_i$ 表示选了 $i$ 个的价值，$f_i$ 是凸函数（上凸：$(i,f_i)$ 形成上凸包，即满足 $f_i-f_{i-1}\geq f_{i+1}-f_i$，如果取 $>$ 就是严格凸。下凸同理）。

解决办法：二分一个 $C$，每选一个物品就额外减去 $C$ 的贡献。若选的个数 $>m$，$C$ 就增大；否则 $C$ 就变少。最后 $C$ 就选 $=m$。其实就是给函数减一个 $y=Cx$，使函数的顶点调整为要算的 $(i,f_i)$。

DP 的凸性很多时候都不太好证，但看见凸优化的征兆后可以打表看看函数是不是凸的。

##### B1

P2619 [国家集训队]Tree I

> 一个 $n$ 个点 $m$ 条边的无向连通图，边有黑白两色，求恰好 $k$ 条白边的最小生成树。
>
> $1\leq n\leq 5\times 10^4$，$1\leq m\leq 10^5$。

WQS 二分。

注意当 $k-1,k,k+1$ 共线时 $k$ 是二分不出来的，这时就钦定权值相同时先选白边还是先选黑边就行。最后算答案时，正常地减去 $k$ 个额外值。

##### B2

CF739E Gosha is hunting（\*3000）

对 A 类求 WQS 二分，B 类求暴力 DP 求 $b$ 个的方案数。时间复杂度 $\mathcal O(n^2\log n)$。

二分 check 时对 B 也 WQS 二分。时间复杂度 $\mathcal O(n\log^2 n)$。

B 类球也可以贪心。选一个 A 类求有 $-mid$ 的额外贡献。那一个球从不选 B 类变成选 B 类的贡献是 $\max(p_i+u_i-p_iu_i-mid,u_i)-\max(p_i-mid,0)$。排序后取最大的 $b$ 个即可。

#### 决策单调性

##### 四边形不等式

以 $\min$ 为例。$\max$ 要反下符号。

函数 $w(x,y)$ 满足 $\forall a\leq b\leq c\leq d$ 都有 $w(a,c)+w(b,d)\leq w(a,d)+w(b,c)$，则 $w$ 满足四边形不等式。即相交优于包含。

另一种定义：$w(l,r-1)+w(l+1,r)\leq w(l,r)+w(l+1,r-1)$。

上推下：$a\gets l$，$b\gets l+1$，$c\gets r-1$，$d\gets r$。

下推上：

- 证明

  可以推出 $w(l,r-1)-w(l,r)\leq w(l+k,r-1)-w(l+k,r)$。

  即 $w(l,r-1)+w(l+k,r)\leq w(l,r)+w(l+k,r-1)$

- 挪 $c$：把 $c=r-1,d=r$ 分开，$w(l,r-1)-w(l+k,r-1)\leq w(l,r)-w(l+k,r)$。

  可以推出 $w(l,r-p)-w(l+k,r-p)\leq w(l,r)-w(l+k,r)$。

  即 $w(l,r-p)+w(l+k,r)\leq w(l,r)+w(l+k,r-p)$。这已经是“相交优于包含”且 $a,b,c,d$ 都能取到的形式了。

##### 优化一维 DP

$f_i=\min\limits_{j<i}\{f_j+w(j,i)\}$，若 $w$ 满足四边形不等式，那么 $f$ 具有决策单调性。

分治：$f_i$ 只受 $f_{i-1}$ 影响，$f_i$ 内部 dp 值不会相互影响。

二分队列：对于 $j<i$，若 $i$ 转移到 $k$ 比 $j$ 优，那么根据决策单调性，由于最优决策点会不断右移，$i$ 转移到 $k\sim n$ 比 $j$ 更优，$j$ 就完全无用了，相当于在时刻 $k$，$i$ 干掉了 $j$。

维护一个单调队列，对 $k$ 从优到劣，队首就是最优转移。

- 弹出队首：若 $q_l$ 在时刻 $i$ 被 $q_{l+1}$ 干掉，就弹出 $q_l$。
- 队尾加入决策点：若 $i$ 干掉 $q_r$ 的最早时刻 $\leq$ $i$ 干掉 $q_{r-1}$ 的最早时刻（时刻可以二分），那么在比 $q_r$ 更优的 $q_{r-1}$ 被干掉前 $q_r$ 自己就被干掉了，于是弹出 $q_r$。

```cpp
//P3515 [POI2011]Lightning Conductor，以 max 为例
double calc(int j,int i){return a[j]+sq[i-j];}
int bound(int x,int y){
	int l=x,r=n,p=n+1;
	while(l<=r){
		int mid=(l+r)/2;
		if(calc(x,mid)>calc(y,mid)) p=mid,r=mid-1;
		else l=mid+1;
	}
	return p;
} 
void solve(double f[N]){
	int l=1,r=0;
	for(int i=1;i<=n;i++){
		while(l<r&&calc(q[l],i)<=calc(q[l+1],i)) l++;	//q[l] 在时刻 i 比 q[l+1] 劣
		f[i]=calc(q[l],i);
		while(l<r&&bound(i,q[r])<=bound(q[r],q[r-1])) r--;	//q[r] 比 q[r-1] 先干掉
		q[++r]=i;
	}
}
```

##### 优化二维 DP

$f_{i,j}=\min\limits_{i\leq k<j}\{f_{i,k}+f_{k+1,j}\}+w(i,j)$，若 $w$ 满足四边形不等式且 $w(b,c)\leq w(a,d)$（包含单调），则 $f$ 满足四边形不等式。

若 $w$ 满足四边形不等式，则对于最优转移点有：$p_{i-1,j}\leq p_{i,j}\leq p_{i,j+1}$。每次循环决策只枚举 $[p_{i,j-1},p_{i+1,j}]$，时间复杂度 $\mathcal O(n^2)$。复杂度证明：枚举量 $\sum p_{i+1,j}-p_{i,j-1}$，发现每个 $p_{l,r}$ 除非某一位是 $n$ 或 $1$，否则分别有一正一负抵消贡献，剩下部分最多 $2n$ 项，每项都 $\leq n$。

### 四、动态 DP

$(\max,+)$ 卷积，有结合律没有交换律。矩阵乘法 $\vec{f_i}=\vec{f_{i-1}}\times w_i$，$w_i$ 是转移矩阵，不一定要一样。

树上：希望把轻儿子合并到重链上去。

##### 小凯的疑惑

nflsoj#268. 【六校联合训练 #9】小凯的疑惑

> 一个长度为 $n$ 的 $01$ 串 $s$，$q$ 次操作：
>
> - `1 l r`：询问将 $[l,r]$ 提出来作为一个二进制数，每次可以把它加上或减去 $2$ 的幂（但不能建成负数），至少多少步可以把它变成 $0$。
> - `2 x y`：$s_x\gets y$。
>
> $n,q\leq 3\times 10^5$。

考虑给出二进制数，如何求最小步数。发现减法产生退位肯定不优，只可能在 $1$ 位置上做减法。

从右往左 DP，设 $f_{i,0/1}$ 表示把后 $i$ 位变成 $0$，加法是否产生进位（右边第 $i\to i+1$）时的最小步数。

- 若当前位为 $0$：

  $f_{i,0}=\min(f_{i-1,0},f_{i-1,1}+1)$（分别：不操作，减一次这位），$f_{i,1}=\min(f_{i-1,0}+2,f_{i-1,1}+1)$（分别：加两次这位，加一次这位）。
  $$
  \begin{bmatrix}f_{i,0}&f_{i,1}\end{bmatrix}
  =
  \begin{bmatrix}f_{i-1,0}&f_{i-1,1}\end{bmatrix}
  \times \begin{bmatrix}0&2\\1&1\end{bmatrix}
  $$

- 若当前位为 $1$：

  $f_{i,0}=\min(f_{i-1,0}+1,f_{i-1,1}+2)$（分别：减一次这位，减两次这位），$f_{i,1}=\min(f_{i-1,0}+1,f_{i-1,1})$（分别：加一次这位，不操作）。
  $$
  \begin{bmatrix}f_{i,0}&f_{i,1}\end{bmatrix}
  =
  \begin{bmatrix}f_{i-1,0}&f_{i-1,1}\end{bmatrix}
  \times
  \begin{bmatrix}1&1\\2&0\end{bmatrix}
  $$

线段树维护修改。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=3e5+5;
int n,q,op,x,y;
char c[N];
struct mat{
	int x[2][2];
	friend mat operator*(mat a,mat b){
		mat c;
		for(int i=0;i<2;i++)
			for(int j=0;j<2;j++){
				c.x[i][j]=1e9; 
				for(int k=0;k<2;k++) c.x[i][j]=min(c.x[i][j],a.x[i][k]+b.x[k][j]);
			}
		return c;
	}
}s[N<<2],a[2]={{0,2,1,1},{1,1,2,0}},tmp;
void build(int p,int l,int r){
	if(l==r){s[p]=a[c[l]-'0'];return ;}
	int mid=(l+r)/2;
	build(p<<1,l,mid);
	build(p<<1|1,mid+1,r);
	s[p]=s[p<<1|1]*s[p<<1];	//从右往左
}
void modify(int p,int l,int r,int pos,int v){
	if(l==r){s[p]=a[v];return ;}
	int mid=(l+r)/2;
	if(pos<=mid) modify(p<<1,l,mid,pos,v);
	else modify(p<<1|1,mid+1,r,pos,v);
	s[p]=s[p<<1|1]*s[p<<1];
}
mat query(int p,int l,int r,int lx,int rx){
	if(l>=lx&&r<=rx) return s[p];
	int mid=(l+r)/2;
	if(rx<=mid) return query(p<<1,l,mid,lx,rx);
	if(lx>mid) return query(p<<1|1,mid+1,r,lx,rx);
	return query(p<<1|1,mid+1,r,lx,rx)*query(p<<1,l,mid,lx,rx); 
}
signed main(){
	scanf("%d%s%d",&n,c+1,&q),build(1,1,n);
	while(q--){
		scanf("%d%d%d",&op,&x,&y);
		if(op==1) tmp=query(1,1,n,x,y),printf("%d\n",min(tmp.x[0][0],tmp.x[0][1]+1));	//!!! 注意不是  tmp.x[0][0]
		else modify(1,1,n,x,y);
	}
	return 0;
}
```

##### 游戏

nflsoj#573. 【六校联合训练 省选 #17】游戏，ZR#2234. 游戏

> Alice 和 Bob 玩游戏。
>
> 一棵 $n$ 个节点的树，每个点上有一些石子。一开始，Alice 可以从任意有石子的点上取走 $1$ 个石子，然后 Bob 只能在 Alice 上次取的位置相邻的有石子的点取走 $1$ 个石子，接下来 Alice 和 Bob 轮流从对方上次取的位置相邻的有石子的点取走 $1$ 个石子，最先不能取任何石子的人输。
>
> $q$ 次修改，每次修改一个节点上的石子数，询问谁必胜。
>
> $1\leq n,q\leq 2\times 10^5$，$0\leq a_i,y\leq 10^9$。

有结论：Bob 能获胜当且仅当树上的石子可以两两配对。

设 $f_i$ 表示以 $i$ 为根的子树，奇数层 - 偶数层的石子个数。$f_x=\sum_{y\in subtree_x}(-1)^{dep_x\oplus dep_y}a_y$。则 Bob 必胜要 $f$ 全 $\geq 0$ 且 $f_1=0$ 才可行。

树链剖分维护 $f$ 即可。先第一遍 dfs 求出未修改前的 $f$，$f_x=a_x-\sum_{y\in son_x}f_y$。然后开两棵线段树，分别维护 $dep_x$ 为奇数/偶数的 $f$。$a_x\gets y$ 时，对于 $x\to 1$ 链上 $dep$ 奇偶性和 $dep_x$ 相同的点 $i$（即奇数层），$f_i\gets f_i+(y-a_x)$，否则就是偶数层，$f_i\gets f_i-(y-a_x)$。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=2e5+5;
int n,q,a[N],x,y,dep[N],fa[N],sz[N],son[N],tim,dfn[N],top[N],o;
ll w[N],cur;
vector<int>v[N];
struct tree{
	ll mn[N<<2],tg[N<<2];
	void upd(int p,ll v){mn[p]+=v,tg[p]+=v;}
	void modify(int p,int l,int r,int lx,int rx,ll v){
		if(l>=lx&&r<=rx) return upd(p,v);
		int mid=(l+r)/2;
		upd(p<<1,tg[p]),upd(p<<1|1,tg[p]),tg[p]=0; 
		if(lx<=mid) modify(p<<1,l,mid,lx,rx,v);
		if(rx>mid) modify(p<<1|1,mid+1,r,lx,rx,v);
		mn[p]=min(mn[p<<1],mn[p<<1|1]);
	} 
}t[2];
void dfs(int x){
	sz[x]=1,w[x]=a[x];
	for(int y:v[x]) if(y!=fa[x]){
		dep[y]=dep[x]+1,fa[y]=x,dfs(y);
		sz[x]+=sz[y],w[x]-=w[y];	//求出初始的 f
		if(sz[y]>sz[son[x]]) son[x]=y; 
	}
} 
void dfs2(int x,int tp){
	dfn[x]=++tim,top[x]=tp;
	t[o=dep[x]&1].modify(1,1,n,dfn[x],dfn[x],w[x]);	//分 dep 为奇数/偶数维护
	t[o^1].modify(1,1,n,dfn[x],dfn[x],1e18);
	if(son[x]) dfs2(son[x],tp);
	for(int y:v[x])
		if(y!=fa[x]&&y!=son[x]) dfs2(y,y);
}
signed main(){
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++) scanf("%d",&a[i]);
	for(int i=1;i<n;i++){
		scanf("%d%d",&x,&y);
		v[x].push_back(y),v[y].push_back(x);
	}
	dep[1]=1,dfs(1),dfs2(1,1),cur=w[1];
	while(q--){
		scanf("%d%d",&x,&y);
		y-=a[x],a[x]+=y,cur+=(o=dep[x]&1)?y:-y;
		for(int i=x;i;i=fa[top[i]]){ 
			t[o].modify(1,1,n,dfn[top[i]],dfn[i],y);	//和 dep[x] 奇偶性相同，奇数层
			t[o^1].modify(1,1,n,dfn[top[i]],dfn[i],-y);	//偶数层
		}
		puts(!cur&&t[0].mn[1]>=0&&t[1].mn[1]>=0?"Yes":"No");
	}
	return 0;
}
```

