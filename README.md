<img style="width: 100%" src="https://raw.githubusercontent.com/daniel-bss/credit-card-clustering/main/img/bg.png">
<h3 align="center" style="margin-top: 12px; font-size: 1.5em">Credit Card Customer Segmentation<h3>

<h2>📄Table of Contents</h2>

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Getting Started](#getting-started)
  - [Clone Repository](#cloning)
  - [Create and Activate Python Virtual Environment](#creating-env)
  - [Installing Required Packages](#installation)
  - [Running The Application](#running)
- [Contributors](#contributors)

<h2 id="problem-statement">🔎Problem Statement</h2>
The banking industry collects insurmountable amounts of customer purchasing data every single day. Often times it is very overwhelming to try and understand large sums of data. However, as mentioned above, it is crucial for any modern business (especially banks) to leverage the data to maximize the decision making process.

<h2 id="solution">💡Solution</h2>
This project will attempt to understand the given data by finding patterns within the dataset using Clustering methodsof Machine Learning. The impact of this analysis will give the user clarity about their current customers and help with the automation of classifying the potential of future incoming customers. This understanding will then translate into specified marketing strategies that could maximize the banks revenue generation.

<h2 id="getting-started">🏃‍♂️Getting Started</h2>
<h3 id="cloning">Clone Repository</h3>
First, easily clone this project into your local machine using this command (make sure you have Git installed).

```
git clone https://github.com/daniel-bss/credit-card-clustering.git
```
<h3 id="creating-env">Create and Activate Python Virtual Environment</h3>

To keep up with the same environment, make sure you have <b>Python 3.9.x</b> installed. Then install `virtualenv` package.

```
pip install virtualenv
```

<br>
Proceed on creating Virtual Environment.

```
virtualenv venv --python="<THE_PATH_TO_YOUR_PYTHON39_EXECUTABLE_FILE>"
```

>Example on Windows:
```
virtualenv venv --python="C:\Users\JohnDoe\AppData\Local\Programs\Python\Python39\python.exe"
```

<br>
Staying still on the root of your project, go activate your Virtual Environment.

```
venv\Scripts\activate
```

<h3 id="installation">Installing Required Packages</h3>

Following the `requirements.txt`, please install the following packages below using this command: `pip install <package>==1.2.3`

```
streamlit==1.5.0
protobuf==3.19.0
click==8.0.4
numpy==1.23.3
pandas==1.5.0
scikit-learn==0.24.1
plotly==5.10.0
```

<h3 id="running">Running The Application</h3>
You have successfully created Virtual Environment with the desired Python version and the required packages. You are now ready to run the Streamlit application using this command.

```
streamlit run main.py
```

<dl>
  <dd>
    <dl>
      <dd>
        <dl>
          <dd>
            <dl>
              <dd>
                <dl>
                  <dd>
                    <dl>
                      <dd>
                        <dl>
                          <dd>
                            <img style="width: 100%;" src="https://raw.githubusercontent.com/daniel-bss/credit-card-clustering/main/img/webapp_2.png">
                          </dd>
                        </dl>
                      </dd>
                    </dl>
                  </dd>
                </dl>
              </dd>
            </dl>
          </dd>
        </dl>
      </dd>
    </dl>
  </dd>
</dl>

<p style="margin-top: 100px;" align="center"><i>(Quick look of the Web App. Created with Streamlit, hosted by Heroku)</i></p>

<h2 id="contributors">👨‍💻Contributors</h2>

- Alexandro Owen Boenardy ([@aowenb](https://github.com/aowenb)) - Business and Data Understanding
- Daniel Bernard Sahala Simamora ([@daniel-bss](https://github.com/daniel-bss)) - Exploratory Data Analysis and Model Deployment
- Maximilian Kevin ([@maxevin](https://github.com/maxevin)) - Evaluation, Conclusion, and Final Deliverables
