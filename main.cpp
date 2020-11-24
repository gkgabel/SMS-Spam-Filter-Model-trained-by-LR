#include <bits/stdc++.h>
#include <fstream>
#include <string>
using namespace std;
int total_size=5171;
int train_size=total_size*.80,feature_vector_size=500;
float modification_rate=0.05;
vector<float> limits={-8,8};
int reg_coefficient_vector_size=feature_vector_size+1;
vector<float> res(reg_coefficient_vector_size);
vector< vector<float> > X(total_size, vector<float>(feature_vector_size));
vector<int> Y(total_size);
int flag=0;
float LogRegLossyFun(vector<float> x){
    float loss_fun=0.0,h=0.0,yhat=0.0;
    for(int i=0;i<train_size;i++)
    {
        yhat=0.0;
        for(int j=0;j<feature_vector_size;j++){
                yhat+=X[i][j]*x[j];

        }
        yhat+=x[feature_vector_size];
        h=1/(1.0+exp(-yhat));
        if(h==1)h=0.9999;
        if(h==0)h=0.0001;
        loss_fun+=(-Y[i]*log(h+0.0))-(1.0-Y[i])*log(1.0-h);
    }
    return loss_fun;
}
float LogRegLossFun(vector<float> x){
    float loss_fun=0.0,h=0.0,yhat=0.0;
    for(int i=0;i<train_size;i++)
    {
        yhat=0.0;
        for(int j=0;j<feature_vector_size;j++){
                yhat+=X[i][j]*x[j];

        }
        yhat+=x[feature_vector_size];
        h=1/(1.0+exp(-yhat));
       // if(h==1)h=0.9999;
       // if(h==0)h=0.0001;
        loss_fun+=pow((Y[i]-h),2);
    }
    return loss_fun;
}
bool comp(vector<float> a,vector<float> b){
    if(a[a.size()-1]>b[b.size()-1])return true;
    return false;
}
vector<float> LogRegObjFun(vector<float> x){
    vector<float> obj_fun(total_size);
    float h=0.0,yhat=0.0;
    for(int i=0;i<total_size;i++)
    {
        yhat=0.0;
        for(int j=0;j<feature_vector_size;j++)yhat+=X[i][j]*x[j];
        yhat+=x[feature_vector_size];
        h=1/(1.0+exp(-yhat));
        obj_fun[i]=h;
    }
    return obj_fun;
}
float rnd(int a=0,int b=1){
    float random = ((float) rand()) / (float) RAND_MAX;
    float t= a+random*(b-a);
    return t;
}
float rndn(float a=0.0,float b=1.0){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed*rnd(0,1));
    std::normal_distribution<double> distribution(a,b);
    float t= distribution(generator);
    //cout<<t<<endl;
    return t;
}
float mod(float x){
    if(x<0)
    return -1*x;
    return x;
}
class kho{
public:
    vector<vector<float>> krill;
    int herd_size;
    float inertial_wt,ct,dmax,vf;
    vector<float> g_best;
    int max_iteration;
    kho(int herd_size,int max_iteration,float inertial_wt,float dmax=0.005,float ct=1)
    {
        this->vf=0.02;
        this->g_best=vector<float>(reg_coefficient_vector_size+1);
        this->ct=ct;
        this->dmax=dmax;
        this->herd_size=herd_size;
        this->inertial_wt=inertial_wt;
        this->max_iteration=max_iteration;
        krill=vector<vector<float>> (herd_size,vector<float>(reg_coefficient_vector_size+1));
        for(int i=0;i<herd_size;i++)
        {
            for(int j=0;j<reg_coefficient_vector_size;j++)
            {
                krill[i][j]=rnd(limits[0],limits[1]);
            }
            //last value is fitness of the individual krill
            krill[i][reg_coefficient_vector_size]=LogRegLossFun(krill[i]);
        }
    }
    float rnd_co(int x,int y,int z){ //generates a random number between x and y except z
        float r=rnd(x,y);
        if(r==z)
        {
            return rnd_co(x,y,z);
        }
        return r;
    }
    float find_distance(vector<float> x,vector<float> y) //to find distance between vectors which is equal to magnitude of the resultant vector
    {
        float rms=0;
        for(int i=0;i<reg_coefficient_vector_size;i++){
            rms+=pow((x[i]-y[i]),2);
        }
        rms=pow(rms,0.5);
        //cout<<rms<<endl;
        return rms;
    }
    vector<float> find_alpha(int i,int best,int worst,int curr_it)  //finds value of alpha=alpha_local+alpha_target
    {
        float ki=krill[i][reg_coefficient_vector_size],kbest=krill[best][reg_coefficient_vector_size],kworst=krill[worst][reg_coefficient_vector_size];
        vector<float> alpha(reg_coefficient_vector_size),alpha_local(reg_coefficient_vector_size),alpha_target(reg_coefficient_vector_size),dist(herd_size);
        float t=0;
        for(int j=0;j<this->herd_size;j++)
        {
            dist[j]=find_distance(krill[i],krill[j]);//distance(xa,xb)=||xa-xb||=sqrt(sigma((xai-xbi)^2))
            t+=dist[j];                              //t=sigma(distance(xa,xi)) for i in range 0 to n
        }
        int ds=t/(5*this->herd_size);//ds=sensing distance
        for(int j=0;j<herd_size;j++)
        {
            float kj=krill[j][reg_coefficient_vector_size];
            if(dist[j]<=ds)
            {
                float ki=krill[i][reg_coefficient_vector_size],kj=krill[j][reg_coefficient_vector_size],kbest=krill[best][reg_coefficient_vector_size],kworst=krill[worst][reg_coefficient_vector_size];
                float kcap=(ki-kj)/(kworst-kbest+0.0001);     //kcap
                for(int k=0;k<reg_coefficient_vector_size;k++)
                {
                    float xcap=(krill[j][k]-krill[i][k])/(dist[j]+0.0001);  //here 0.001 is epsilon which prevents singularity
                    alpha_local[k]+=kcap*xcap;
                }
            }
        }
        float cbest=2*(rnd(0,1)+(0.0+curr_it)/(0.0+this->max_iteration));
        for(int k=0;k<reg_coefficient_vector_size;k++)
        {

                float kcap=(ki-kbest)/(kworst-kbest+0.0001);
                float xcap=(krill[best][k]-krill[i][k])/(dist[best]+0.0001);//here 0.001 is phi which prevents singularity
                alpha_target[k]=cbest*kcap*xcap;
       }
        for(int k=0;k<reg_coefficient_vector_size;k++)
        {
                alpha[k]=alpha_target[k]+alpha_local[k];
        }
        return alpha;
    }
    void find_fit()
    {
        time_t start, finish;
        time(&start);
        //lagrangian model
        //grad=dx/dt=N+F+D whwere N is movement induced by other krill, F is foraging speed and D is random diffusion
        vector<vector<float>> N(herd_size,vector<float> (reg_coefficient_vector_size)),F,D;
        F=N;
        D=N;
        float Nmax=0.01;
        int best=0,worst=0;
        for(int i=0;i<herd_size;i++)
        {
            if(krill[i][reg_coefficient_vector_size]>krill[worst][reg_coefficient_vector_size])worst=i;
            if(krill[i][reg_coefficient_vector_size]<krill[best][reg_coefficient_vector_size])best=i;
        }
        g_best=krill[best];
        for(int it=1;it<=max_iteration;it++)
        {
            float kbest=krill[best][reg_coefficient_vector_size],kworst=krill[worst][reg_coefficient_vector_size];
            //finding motion induced due to other krills
            for(int i=0;i<herd_size;i++)
            {
                vector<float> alpha=find_alpha(i,best,worst,it);
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    N[i][j]=Nmax*alpha[j]+inertial_wt*N[i][j];
                }
            }
            //foraging motion
            vector<float> food(reg_coefficient_vector_size+1);
            for(int j=0;j<reg_coefficient_vector_size;j++)
            {
                float x=0,y=0;
                for(int i=0;i<herd_size;i++)
                {
                    x+=krill[i][j]/krill[i][reg_coefficient_vector_size];
                    y+=1/krill[i][reg_coefficient_vector_size];
                }
                food[j]=x/y;
            }
            food[reg_coefficient_vector_size]=LogRegLossFun(food);
            float cfood=2*(1-it/(0.0+max_iteration));
            for(int i=0;i<herd_size;i++)
            {
                float ki=krill[i][reg_coefficient_vector_size],kfood=food[reg_coefficient_vector_size];
                float kcap=(ki-kfood)/(kworst-kbest),kcapbest=(ki-kbest)/(kworst-kbest+0.0001);
                vector<float> xcap(reg_coefficient_vector_size),xcapbest,betafood,betabest;
                betafood=xcap;
                betabest=xcap;
                xcapbest=xcap;
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    xcapbest[j]=(krill[best][j]-krill[i][j])/(0.0001+find_distance(krill[best],krill[i]));
                    xcap[j]=(food[j]-krill[i][j])/(0.0001+find_distance(food,krill[i]));
                    betafood[j]=cfood*kcap*xcap[j];
                    betabest[j]=kcapbest*xcapbest[j];
                }
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    F[i][j]=vf*(betafood[j]+betabest[j])+inertial_wt*F[i][j];
                }

            }
            //physical diffusion
            float itf=(1-it/(0.0+max_iteration));
            for(int i=0;i<herd_size;i++)
            {
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    float delta=rnd(-1,1);
                    D[i][j]=dmax*itf*delta;
                }
            }
            vector<vector<float>> new_krill=krill;
            //summing up, we get dx/dt=N+F+D
            //and x(t+del(t))=x(t)+del(t)*dx/dt)
            for(int i=0;i<herd_size;i++)
            {
                vector<float> new_sol(reg_coefficient_vector_size+1);
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    float delta_t=ct*(limits[1]-limits[0]);
                    //cout<<i<<" "<<j<<" "f<<N[i][j]<<" "<<F[i][j]<<" "<<D[i][j]<<endl;
                    new_sol[j]=krill[i][j]+delta_t*(N[i][j]+F[i][j]+D[i][j]);
                    if(new_sol[j]<limits[0])new_sol[j]=limits[0];
                    if(new_sol[j]>limits[1])new_sol[j]=limits[1];
                }
                new_sol[reg_coefficient_vector_size]=LogRegLossFun(new_sol);
                if(krill[i][reg_coefficient_vector_size]>new_sol[reg_coefficient_vector_size])
                    krill[i]=new_sol;
            }
            for(int i=0;i<herd_size;i++)
            {
                if(krill[i][reg_coefficient_vector_size]>krill[worst][reg_coefficient_vector_size])worst=i;
                if(krill[i][reg_coefficient_vector_size]<krill[best][reg_coefficient_vector_size])best=i;
            }
            //crossover operation binomial
            for(int i=0;i<herd_size;i++)
            {
                float ki=krill[i][reg_coefficient_vector_size],kbest=krill[best][reg_coefficient_vector_size],kworst=krill[worst][reg_coefficient_vector_size];
                float cr=0.2*(ki-kbest)/(kworst-kbest+0.0001);
                int p=rnd_co(0,herd_size-1,i);
                for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    if(rnd(0,1)<cr)
                    {
                        krill[i][j]=krill[p][j];
                    }
                }
                krill[i][reg_coefficient_vector_size]=LogRegLossFun(krill[i]);
                //if(new_krill[i][reg_coefficient_vector_size]<krill[i][reg_coefficient_vector_size])krill[i]=new_krill[i];
            }
            if(krill[best][reg_coefficient_vector_size]<g_best[reg_coefficient_vector_size])
            {
                g_best=krill[best];
            }
            else
            {
                krill[worst]=g_best;
            }
            for(int i=0;i<herd_size;i++)
            {
                if(krill[i][reg_coefficient_vector_size]>krill[worst][reg_coefficient_vector_size])worst=i;
                if(krill[i][reg_coefficient_vector_size]<krill[best][reg_coefficient_vector_size])best=i;
            }
            time(&finish);
           cout<<"iteration no. "<<it<<":"<<setprecision(5)<<g_best[reg_coefficient_vector_size]<<setw(20)<<finish-start<<endl;
        }
        res=g_best;
    }

};
class abc{
public:
    //variables
    std::vector<std::vector<float> > boundaries,agents,food;
    std::vector<int>trial_counter;
    std::vector<float> fit;
    int colony_size,scouts,iterations,counter_limit,employed_onlookers_count,scout_status;
    float max_fit,modification_rate;
    std::vector<float> best_food_source;
    //functions
    abc(int colony_size,int limit,int iterations,float modification_rate=0.05){
        //cout<<"abc::abc(int colony_size,int scouts,int iterations)\n";
        this->modification_rate=modification_rate;
        this->colony_size=colony_size;
        this->iterations=iterations;
        this->scouts=0;
        this->counter_limit=limit;
        this->max_fit=0.0;
        this->best_food_source=vector<float>(reg_coefficient_vector_size);
        this->employed_onlookers_count = colony_size/2;
        this->trial_counter=vector<int> (employed_onlookers_count);
        this->food=vector<vector<float> > (employed_onlookers_count,vector<float>  (reg_coefficient_vector_size));
        for(int i=0;i<this->employed_onlookers_count;i++)
            for(int j=0;j<reg_coefficient_vector_size;j++)
                {
                    this->food[i][j]=rnd(limits[0],limits[1]);
                }
        for(int i=0;i<this->food.size();i++)
        {
            fit.push_back(calculate_fit(this->food[i]));
        }
        for(int j=0;j<this->food.size();j++)
        {
                if(max_fit<this->fit[j])
                {
                    max_fit=this->fit[j];
                    this->best_food_source=food[j];
                }
        }
    }
    void evaluate_neighbor(int current_position,int partner_position){               //cout<<"void abc::evaluate_neighbor(int current_position,int partner_position)\n";
                std::vector<float> neighbor=food[current_position];
                //modified abc with local search operator to improve convergence rate with modification rate MR
                for (int j=0;j<reg_coefficient_vector_size;j++)
                {
                    float xj_new;
                    if (rnd()<modification_rate)
                        {
                            xj_new = this->food[current_position][j] + rnd(-1,1)*(this->food[current_position][j] - this->food[partner_position][j]);
                        }
                    else xj_new = this->food[current_position][j];
                    //Check boundary
                    if (xj_new < limits[0])xj_new = limits[0];
                    else if (xj_new > limits[1])xj_new=limits[1];
                    //Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
                    neighbor[j] =xj_new;
                }
                float neighbor_fit = this->calculate_fit(neighbor);
                //Greedy selection
                if (neighbor_fit > this->fit[current_position])
                {
                    this->food[current_position]=neighbor;
                    this->fit[current_position] = neighbor_fit;
                    this->trial_counter[current_position] = 0;
                    if(max_fit<this->fit[current_position]){
                            max_fit=fit[current_position];
                            best_food_source=food[current_position];
                    }
                }
                else this->trial_counter[current_position]++;
    }

    float calculate_fit(vector<float> evaluated_position){
        //cout<<"float abc::calculate_fit(vector<float> evaluated_position){\n";
        //eq. (2) [2] (Convert "cost function" to "fit function")
        float cost = LogRegLossFun(evaluated_position);
        float fit_value;
        if (cost < 0)fit_value = (1 - cost);
        else fit_value=(1/(1 + cost));
        return fit_value;
    }
    void food_source_dance(int index){
        //cout<<"void abc::food_source_dance(int index){\n";
        int d;
        while(1) //Criterion from [1] getting another food source at random
        {
            d = rnd(0,this->employed_onlookers_count-1);
            if(d != index)
                break;
        }
        this->evaluate_neighbor(index,d);
    }
    void find_fit(){
        vector<float> lr_val(iterations);
        //cout<<"void abc::find_fit(){\n";
        for(int j=0;j<colony_size/2;j++)this->max_fit=max(max_fit,this->fit[j]);
        for(int it=1;it<=iterations;it++){
            //--> Employer bee phase <--
            //Generate and evaluate a neighbor point to every food source
            for(int j=0;j<employed_onlookers_count;j++)food_source_dance(j);
            vector<float> onlooker_probability(colony_size/2);
            for(int j=0;j<colony_size/2;j++)onlooker_probability[j]=0.9*(fit[j]/max_fit)+0.1;
            //--> Onlooker bee phase <--
            //Based in probability, generate a neighbor point and evaluate again some food sources
            //Same food source can be evaluated multiple times
            int p = 0;//Onlooker bee index
            int i = 0; //Food source index
            while (p < employed_onlookers_count){
                if (rnd(0,1) < onlooker_probability[i]){
                        p++;
                    food_source_dance(i);
                }
                if (i < (this->employed_onlookers_count-1))
                    i++;
                else
                    i=0;
            }
            //--> Memorize best solution <--
            for(int j=0;j<colony_size/2;j++)
            {
                if(max_fit<fit[j])
                {
                    max_fit=fit[j];
                    best_food_source=food[j];
                }
            }
            //--> Scout bee phase <--
            //Generate up to one new food source that does not improve over scout_limit evaluation tries
            float max_trial_counter=0;
            if (trial_counter[i]>this->counter_limit){
                    if(max_trial_counter<trial_counter[i])
                    {
                        max_trial_counter=trial_counter[i];
                    }
            }
            if(max_trial_counter>counter_limit)
            for(int i=0;i<colony_size/2;i++)
            {
                if (trial_counter[i]==max_trial_counter){
                    cout<<trial_counter[i]<<endl;
                    trial_counter[i]=0;
                    for(int j=0;j<reg_coefficient_vector_size;j++)
                        this->food[i][j]=rnd(limits[0],limits[1]); //Replace food source
                    fit[i]=calculate_fit(this->food[i]);
                }
                this->scout_status++;
            }

            lr_val[it-1]=LogRegLossFun(this->best_food_source);
            cout<<"iteration:"<<setprecision(4)<<it<<"   "<<lr_val[it-1]<<endl;
        }
    res=this->best_food_source;
    /*ofstream opfile;
    opfile.open("abc_output.txt");
    if (opfile.is_open())
    {
        for(int i=0;i<res.size();i++)
        {
            cout<<res[i]<<" ";
            opfile<<res[i]<<" ";
        }
        opfile.close();
    }
            ofstream lr_res;
        lr_res.open("abc_lr_res.txt");
        if (lr_res.is_open())
        {
            for(int i=0;i<lr_val.size();i++)
            {
                cout<<lr_val[i]<<" ";
                lr_res<<lr_val[i]<<" ";
            }
            lr_res.close();
        }
    */
    }
};
class cso{
public:
    //variables
    std::vector<std::vector<float> > nest;
    std::vector<float> fit,boundaries,best_nest;
    int population,dimension,iterations;
    float pa,beta,sigma;
    //functions
    cso(vector<float> boundaries,int population,int iterations,float pa,int dimension=501){
        this->boundaries=boundaries;
        this->population=population;
        this->dimension=dimension;
        this->pa=pa;
        this->iterations=iterations;
        this->nest=vector<vector<float>> (population,vector<float> (dimension+1));
        this->beta=1.5;
        this->sigma=0.6966;
        for(int i=0;i<population;i++)
        {
            for(int j=0;j<dimension;j++)
            {
                nest[i][j]=boundaries[0]+rnd(0,1)*(boundaries[1]-boundaries[0]);
            }
            nest[i][dimension]=calculate_fit(nest[i]);
        }
        this->best_nest=vector<float>(dimension+1);
    }
    float calculate_fit(std::vector<float> nest_position){
        float fitness=LogRegLossFun(nest_position);
        return fitness;
    }
    void find_fit(){
        time_t start, finish;
        time(&start);
        vector<float> lr_val(iterations);
        sort(nest.begin(),nest.end(),comp);
        vector<float> best_sol=nest[0];
        float best_fit=nest[0][dimension];
        for(int it=0;it<iterations;it++)
        {
            //new solution generation
            for(int i=0;i<population;i++)
            {
                double u,v;
                vector<float> new_sol=nest[i];
                for(int j=0;j<dimension;j++)

                {
                    u=sigma*rndn(0,100);
                    v=rndn(0,100);
                    double s=u/pow(mod(v),1.0/beta);
                    new_sol[j]=nest[i][j]+0.01*s*(nest[i][j]-nest[0][j]);
                    if(new_sol[j]>boundaries[1])new_sol[j]=boundaries[1];
                    if(new_sol[j]<boundaries[0])new_sol[j]=boundaries[0];
                }
                //greedy selection
                new_sol[dimension]=calculate_fit(new_sol);
                if(nest[i][dimension]>new_sol[dimension])
                {
                    nest[i]=new_sol;
                    if(new_sol[dimension]<best_fit)
                    {
                        best_fit=new_sol[dimension];
                        best_sol=new_sol;
                    }
                }

            }
            //nest abandon
            for(int i=0;i<population;i++)
            {

                vector<float> new_sol=nest[i];
                if(rnd(0,1)<pa)
                {
                int d1=rnd(0,population-1),d2=rnd(0,population-1);
                    for(int j=0;j<dimension;j++)
                    {
                            new_sol[j]=nest[i][j]+rnd(0,1)*(nest[d1][j]-nest[d2][j]);
                            //checking bounds--
                            if(new_sol[j]>boundaries[1])new_sol[j]=boundaries[1];
                            if(new_sol[j]<boundaries[0])new_sol[j]=boundaries[0];
                    }
                    //greedy selection
                    new_sol[dimension]=calculate_fit(new_sol);
                    if(nest[i][dimension]>new_sol[dimension])
                    {
                        nest[i]=new_sol;
                        if(new_sol[dimension]<best_fit)
                        {
                            best_fit=new_sol[dimension];
                            best_sol=new_sol;
                        }
                    }
                }

            }
            time(&finish);
            lr_val[it]=best_fit;
            cout<<"iteration "<<it<<":"<<setprecision(5)<<lr_val[it]<<setw(20)<<finish-start<<endl;
        }
       /* ofstream optfile;
        optfile.open("cso_output.txt");
        if (optfile.is_open())
        {
            for(int i=0;i<res.size();i++)
            {
                cout<<best_sol[i]<<" ";
                optfile<<best_sol[i]<<" ";
            }
            optfile.close();
        }
        ofstream lrres;
        lrres.open("cso_lr_res.txt");
        if (lrres.is_open())
        {
            for(int i=0;i<lr_val.size();i++)
            {
                cout<<lr_val[i]<<" ";
                lrres<<lr_val[i]<<" ";
            }
            lrres.close();
        }*/
        res=best_sol;
    }
    void mod_find_fit(){
        float abd_ratio=0.75,phi=(pow(5,0.5)-1)/2;
        vector<float> lr_val(iterations);
        int g=1;
        for(int it=0;it<iterations;it++)
        {
            g++;
            sort(nest.begin(),nest.end(),comp);
            vector<float> best_sol(dimension);
            float best_fit=nest[0][dimension];

            for(int i=(1-abd_ratio)*population;i<population;i++)
            {
                float malpha=.1/pow(g,0.5);
                double u,v;
                vector<float> new_sol=nest[i];
                for(int j=0;j<dimension;j++)
                {
                    u=sigma*rndn(0,100);
                    v=rndn(0,100);
                    double s=u/pow(mod(v),1.0/beta);
                    //cout<<j<<" "<<u<<" "<<v<<" "<<s<<endl;
                    new_sol[j]=nest[i][j]+malpha*s*(nest[i][j]-nest[0][j]);
                    //checking bounds
                    if(new_sol[j]>boundaries[1])new_sol[j]=boundaries[1];
                    if(new_sol[j]<boundaries[0])new_sol[j]=boundaries[0];
                }
                //greedy selection
                new_sol[dimension]=calculate_fit(new_sol);
                if(nest[i][dimension]>new_sol[dimension])
                {
                    nest[i]=new_sol;
                }

            }
            //nest abandon
            for(int i=0;i<(1-abd_ratio)*population;i++)
            {

                vector<float> new_sol=nest[i];

                int new_nest=rnd(0,(1-abd_ratio)*population-1);
                if(new_nest==i)
                {
                    float malpha=.1/pow(g,2);
                    for(int j=0;j<dimension;j++)
                    {
                            float u=sigma*rndn(0,100);
                            float v=rndn(0,100);
                            double s=u/pow(mod(v),1.0/beta);
                            new_sol[j]=nest[i][j]+malpha*s*(nest[i][j]-nest[0][j]);
                    //checking bounds
                            if(new_sol[j]>boundaries[1])new_sol[j]=boundaries[1];
                            if(new_sol[j]<boundaries[0])new_sol[j]=boundaries[0];
                    }
                    //greedy selection
                    int rnd_nest=rnd(0,population-1);
                    new_sol[dimension]=calculate_fit(new_sol);
                    if(calculate_fit(nest[rnd_nest])>new_sol[dimension])
                    {
                        nest[rnd_nest]=new_sol;
                    }
                }
                else{
                    for(int j=0;j<dimension;j++)
                    {
                        float dx;
                        if(nest[i][dimension]<nest[new_nest][dimension])
                        {
                            dx=(nest[i][j]-nest[new_nest][j])/phi;
                            new_sol[j]=nest[new_nest][j]-dx;
                        }
                        else
                        {
                            dx=(nest[i][j]-nest[new_nest][j])/phi;
                            new_sol[j]=nest[i][j]+dx;
                        }

                    }
                    int rnd_nest=rnd(0,population-1);
                    new_sol[dimension]=calculate_fit(new_sol);
                    if(nest[rnd_nest][dimension]>new_sol[dimension])
                    {
                        nest[rnd_nest]=new_sol;
                    }

                }
            }
            for(int i=0;i<population;i++)
            {

                vector<float> new_sol=nest[i];
                if(rnd(0,1)<pa)
                {
                int d1=rnd(0,population-1),d2=rnd(0,population-1);
                    for(int j=0;j<dimension;j++)
                    {
                            new_sol[j]=nest[i][j]+rnd(0,1)*(nest[d1][j]-nest[d2][j]);
                            //checking bounds--
                            if(new_sol[j]>boundaries[1])new_sol[j]=boundaries[1];
                            if(new_sol[j]<boundaries[0])new_sol[j]=boundaries[0];
                    }
                    //greedy selection
                    new_sol[dimension]=calculate_fit(new_sol);
                    if(nest[i][dimension]>new_sol[dimension])
                    {
                        nest[i]=new_sol;
                        if(new_sol[dimension]<best_fit)
                        {
                            best_fit=new_sol[dimension];
                        }
                    }
                }

            }
            lr_val[it]=best_fit;
            cout<<"iteration "<<it<<":"<<lr_val[it]<<endl;
        }
        ofstream optfile;
        res=best_nest;
        /*optfile.open("cso_output.txt");
        if (optfile.is_open())
        {
            for(int i=0;i<res.size();i++)
            {
                cout<<best_nest[i]<<" ";
                optfile<<best_nest[i]<<" ";
            }
            optfile.close();
        }
        ofstream lrres;
        lrres.open("cso_lr_res.txt");
        if (lrres.is_open())
        {
            for(int i=0;i<lr_val.size();i++)
            {
                cout<<lr_val[i]<<" ";
                lrres<<lr_val[i]<<" ";
            }
            lrres.close();
        }*/
    }
};
int main () {
    srand (time(NULL));
    /* initialize random seed: */
    float temp;
    ifstream myfile ("sparse_array_enron500.txt");
    if (myfile.is_open())
    {
        int i=0,j=0;
        for(i=0;i<total_size;i++)
        {
            for(j=0;j<feature_vector_size;j++)
            {
                myfile>>temp;
                X[i][j]=temp;
            }
        }
    myfile.close();
    }
    else cout << "Unable to open file 1";
    ifstream mf ("Y_1.txt");
    if (mf.is_open())
    {
        float d;
        for(int i=0;i<total_size;i++)
        {
            mf>>d;
            Y[i]=d;
        }
    mf.close();
    }
    else cout << "Unable to open file 2";
    vector<float> x(reg_coefficient_vector_size);
    /*int a[3]={100,500,1000};
    for(int i=0;i<3;i++)
    {
        ::feature_vector_size=a[i];
        ::reg_coefficient_vector_size=a[i]+1;
        abc A=abc(40,100,1000);
        A.find_fit();
    }*/
    //cso C=cso({-5,5},10,1000,.5,reg_coefficient_vector_size);
    //C.find_fit();

    for(int col=40;col<=40;col+=20)
    {
        for(int p=0;p<100;p++)
        {
            abc K=abc(40,100,2000,0.1);
            K.find_fit();
            vector<float> fval=LogRegObjFun(res);
            int fn=0,fp=0,tn=0,tp=0;
            for(int i=train_size;i<total_size;i++)
            {
                if(fval[i]>.5)
                {
                    if(Y[i])tp++;
                    else fp++;
                }
                else
                {
                    if(Y[i])fn++;
                    else tn++;
                }
            }
            cout<<"\nWe are bee\n"<<p<<endl;
            cout<<"\naccuracy="<<(tn+tp)/(tn+tp+fn+fp+0.0);
            cout<<"\nSensitivity/Recall="<<tp/(tp+fn+0.0);
            cout<<"\nspecificity="<<tn/(tn+fp+0.0);
            cout<<"\nprecision="<<tp/(tp+fp+0.0);
            float precision=tp/(tp+fp+0.0),recall=tp/(tp+fn+0.0);
            cout<<"\nF1 score="<<2*(precision*recall)/(precision+recall);
            ofstream evp;
            evp.open("eval_param_abc_enron.txt",ios_base::app);
            evp<<fn<<" "<<fp<<" "<<tn<<" "<<tp<<"\n";
            evp.close();
        }

    }
    return 0;
}


