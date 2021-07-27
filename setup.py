from setuptools import setup, find_packages


setup(name='qmbmodels',
      version='1.1.0',
      description='A module for calculations with 1D quantum hamiltonians',
      url='https://github.com/JanSuntajs/qmbmodels',
      author='Jan Suntajs',
      author_email='Jan.Suntajs@ijs.si',
      license='MIT',
      packages=find_packages(),
      install_requires=[('ham1d @ git+https://github.com/JanSuntajs/'
                         'ham1d@v1.2.5#egg=ham1d'), ],
      # dependency_links=[('qmbhttps://github.com/JanSuntajs/'
      #                   'ham1d/tarball/'
      #                   'master/#egg=ham1d-1.2.5')],
      zip_safe=False)
